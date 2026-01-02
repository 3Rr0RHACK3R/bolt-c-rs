use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bolt_core::Backend;
use bolt_core::backend::{AddOp, FillOp};
use bolt_core::dtype::{Float, NativeType};
use bolt_core::error::{Error, Result};

use crate::Tensor;

mod ops;
pub(crate) mod utils;

pub(crate) use ops::{
    AbsBackward, AddBackward, BroadcastToBackward, CosBackward, DivBackward, ExpBackward,
    LogBackward, MatmulBackward, MaxBackward, MeanBackward, MinBackward, MulBackward, NegBackward,
    PowBackward, ProdBackward, ReluBackward, ReshapeBackward, SigmoidBackward, SinBackward,
    SqrtBackward, SqueezeAllBackward, SqueezeAxisBackward, SubBackward, SumBackward, TanhBackward,
    TransposeBackward, UnsqueezeBackward,
};
pub use utils::canonical_axes;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_tensor_id() -> TensorId {
    let id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
    TensorId(id)
}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

pub struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

pub fn no_grad() -> NoGradGuard {
    let prev = GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(false);
        prev
    });
    NoGradGuard { prev }
}

pub struct Grads<B, D>
where
    B: Backend,
    D: Float,
{
    leaf: HashMap<TensorId, Tensor<B, D>>,
}

impl<B, D> Grads<B, D>
where
    B: Backend,
    D: Float,
{
    pub fn wrt(&self, tensor: &Tensor<B, D>) -> Option<&Tensor<B, D>> {
        self.leaf.get(&tensor.tensor_id())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BackwardOptions {
    pub retain_graph: bool,
}

pub(crate) struct InputEdge<B, D>
where
    B: Backend,
    D: NativeType,
{
    pub id: TensorId,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<GradFn<B, D>>>,
}

pub(crate) struct GradFn<B, D>
where
    B: Backend,
    D: NativeType,
{
    pub output: TensorId,
    pub inputs: Vec<InputEdge<B, D>>,
    pub op: Box<dyn BackwardOp<B, D>>,
    // CRITICAL: saved tensors must be detached to avoid Arc cycles
    pub saved_tensors: Vec<Tensor<B, D>>,
}

pub(crate) struct BackwardContext<'a, B, D>
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    saved: &'a [Tensor<B, D>],
}

impl<'a, B, D> BackwardContext<'a, B, D>
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    fn new(saved: &'a [Tensor<B, D>]) -> Self {
        Self { saved }
    }

    pub fn saved(&self, idx: usize) -> &Tensor<B, D> {
        &self.saved[idx]
    }
}

pub(crate) trait BackwardOp<B, D>: Send + Sync
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>>;
    fn name(&self) -> &'static str;
}

#[derive(Clone)]
pub(crate) struct AutogradMeta<B, D>
where
    B: Backend,
    D: NativeType,
{
    pub origin: TensorId,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<GradFn<B, D>>>,
}

impl<B, D> AutogradMeta<B, D>
where
    B: Backend,
    D: NativeType,
{
    pub fn new() -> Self {
        Self {
            origin: next_tensor_id(),
            requires_grad: false,
            grad_fn: None,
        }
    }
}

impl<B, D> Default for AutogradMeta<B, D>
where
    B: Backend,
    D: NativeType,
{
    fn default() -> Self {
        Self::new()
    }
}

fn collect_nodes<B, D>(root: &Arc<GradFn<B, D>>) -> Vec<Arc<GradFn<B, D>>>
where
    B: Backend,
    D: NativeType,
{
    let mut visited: HashSet<TensorId> = HashSet::new();
    let mut order: Vec<Arc<GradFn<B, D>>> = Vec::new();
    let mut stack: Vec<(Arc<GradFn<B, D>>, bool)> = vec![(Arc::clone(root), false)];

    while let Some((node, children_processed)) = stack.pop() {
        if children_processed {
            order.push(node);
        } else if !visited.contains(&node.output) {
            visited.insert(node.output);
            stack.push((Arc::clone(&node), true));

            for input in node.inputs.iter().rev() {
                if let Some(parent) = &input.grad_fn
                    && !visited.contains(&parent.output)
                {
                    stack.push((Arc::clone(parent), false));
                }
            }
        }
    }

    order
}

pub fn backward<B, D>(loss: &Tensor<B, D>) -> Result<Grads<B, D>>
where
    B: Backend + AddOp<D> + FillOp<D> + 'static,
    D: Float + 'static,
{
    backward_with_options(loss, BackwardOptions::default())
}

pub fn backward_with_options<B, D>(loss: &Tensor<B, D>, _options: BackwardOptions) -> Result<Grads<B, D>>
where
    B: Backend + AddOp<D> + FillOp<D> + 'static,
    D: Float + 'static,
{
    if !loss.requires_grad_enabled() {
        return Err(Error::OpError(
            "cannot compute backward: loss tensor does not require gradient".into(),
        ));
    }

    let _ng = no_grad();

    let Some(root_grad_fn) = loss.grad_fn() else {
        let seed = if loss.numel() == 1 {
            let backend = loss.backend();
            Tensor::full(&backend, &[], D::one())?
        } else {
            Tensor::ones_like(loss)?
        };
        let mut leaf = HashMap::new();
        leaf.insert(loss.tensor_id(), seed);
        return Ok(Grads { leaf });
    };

    let nodes = collect_nodes(&root_grad_fn);
    
    if nodes.is_empty() {
        return Err(Error::OpError(
            "cannot compute backward: computational graph is empty".into(),
        ));
    }

    let seed = if loss.numel() == 1 {
        let backend = loss.backend();
        Tensor::full(&backend, &[], D::one())?
    } else {
        Tensor::ones_like(loss)?
    };

    let mut grads: HashMap<TensorId, Tensor<B, D>> = HashMap::new();
    grads.insert(loss.tensor_id(), seed);

    let produced: HashSet<TensorId> = nodes.iter().map(|n| n.output).collect();
    let mut leaf_required: HashSet<TensorId> = HashSet::new();
    for node in &nodes {
        for inp in &node.inputs {
            if inp.requires_grad && !produced.contains(&inp.id) {
                leaf_required.insert(inp.id);
            }
        }
    }

    for node in nodes.into_iter().rev() {
        let Some(grad_output) = grads.remove(&node.output) else {
            continue;
        };

        let ctx = BackwardContext::new(&node.saved_tensors);
        let input_grads = node.op.backward(&grad_output, &ctx)?;

        if input_grads.len() != node.inputs.len() {
            return Err(Error::OpError(format!(
                "backward op {} returned {} grads for {} inputs",
                node.op.name(),
                input_grads.len(),
                node.inputs.len()
            )));
        }

        for (inp, gopt) in node.inputs.iter().zip(input_grads.into_iter()) {
            if !inp.requires_grad {
                continue;
            }
            let Some(g) = gopt else {
                continue;
            };
            insert_or_accumulate(&mut grads, inp.id, g)?;
        }
    }

    let leaf = grads
        .into_iter()
        .filter(|(id, _)| leaf_required.contains(id))
        .collect();
    Ok(Grads { leaf })
}

fn insert_or_accumulate<B, D>(
    map: &mut HashMap<TensorId, Tensor<B, D>>,
    key: TensorId,
    grad: Tensor<B, D>,
) -> Result<()>
where
    B: Backend + AddOp<D>,
    D: Float,
{
    match map.remove(&key) {
        None => {
            map.insert(key, grad);
        }
        Some(existing) => {
            let backend = existing.backend();
            let parts = backend.add(
                existing.storage(),
                grad.storage(),
                existing.layout(),
                grad.layout(),
            )?;
            let acc = Tensor::from_parts(backend, parts.storage, parts.layout);
            map.insert(key, acc);
        }
    }
    Ok(())
}

pub(crate) fn create_unary_grad_fn<B, D>(
    output_id: TensorId,
    input: &Tensor<B, D>,
    op: Box<dyn BackwardOp<B, D>>,
    saved_tensors: Vec<Tensor<B, D>>,
) -> Option<Arc<GradFn<B, D>>>
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    if !D::SUPPORTS_GRAD || !grad_enabled() || !input.requires_grad_enabled() {
        return None;
    }

    let saved_tensors: Vec<_> = saved_tensors.into_iter().map(|t| t.detach()).collect();

    Some(Arc::new(GradFn {
        output: output_id,
        inputs: vec![InputEdge {
            id: input.tensor_id(),
            requires_grad: input.requires_grad_enabled(),
            grad_fn: input.grad_fn(),
        }],
        op,
        saved_tensors,
    }))
}

pub(crate) fn create_binary_grad_fn<B, D>(
    output_id: TensorId,
    lhs: &Tensor<B, D>,
    rhs: &Tensor<B, D>,
    op: Box<dyn BackwardOp<B, D>>,
    saved_tensors: Vec<Tensor<B, D>>,
) -> Option<Arc<GradFn<B, D>>>
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    if !D::SUPPORTS_GRAD || !grad_enabled() {
        return None;
    }
    
    if !lhs.requires_grad_enabled() && !rhs.requires_grad_enabled() {
        return None;
    }

    let saved_tensors: Vec<_> = saved_tensors.into_iter().map(|t| t.detach()).collect();

    Some(Arc::new(GradFn {
        output: output_id,
        inputs: vec![
            InputEdge {
                id: lhs.tensor_id(),
                requires_grad: lhs.requires_grad_enabled(),
                grad_fn: lhs.grad_fn(),
            },
            InputEdge {
                id: rhs.tensor_id(),
                requires_grad: rhs.requires_grad_enabled(),
                grad_fn: rhs.grad_fn(),
            },
        ],
        op,
        saved_tensors,
    }))
}
