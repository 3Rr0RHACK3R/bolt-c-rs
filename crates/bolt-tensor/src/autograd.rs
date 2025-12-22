use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

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
    PowBackward, ProdBackward, ReluBackward, ReshapeBackward, SinBackward, SqrtBackward,
    SqueezeAllBackward, SqueezeAxisBackward, SubBackward, SumBackward, TanhBackward,
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
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
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

struct Input {
    id: TensorId,
    requires_grad: bool,
}

struct BackwardEntry<B, D>
where
    B: Backend,
    D: NativeType,
{
    op: Box<dyn BackwardOp<B, D>>,
    saved_tensors: Vec<Tensor<B, D>>,
}

struct Node<B, D>
where
    B: Backend,
    D: NativeType,
{
    output: TensorId,
    inputs: Vec<Input>,
    backward: BackwardEntry<B, D>,
}

struct TapeArena<B, D>
where
    B: Backend,
    D: NativeType,
{
    nodes: Vec<Node<B, D>>,
}

impl<B, D> TapeArena<B, D>
where
    B: Backend,
    D: NativeType,
{
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn record(&mut self, output: TensorId, inputs: Vec<Input>, backward: BackwardEntry<B, D>) {
        self.nodes.push(Node {
            output,
            inputs,
            backward,
        });
    }

    fn take_nodes(&mut self) -> Vec<Node<B, D>> {
        std::mem::take(&mut self.nodes)
    }
}

struct Runtime {
    tapes: HashMap<(TypeId, TypeId), Box<dyn Any>>,
}

impl Runtime {
    fn new() -> Self {
        Self {
            tapes: HashMap::new(),
        }
    }

    fn tape_mut<B, D>(&mut self) -> &mut TapeArena<B, D>
    where
        B: Backend + 'static,
        D: NativeType + 'static,
    {
        let key = (TypeId::of::<B>(), TypeId::of::<D>());
        self.tapes
            .entry(key)
            .or_insert_with(|| Box::new(TapeArena::<B, D>::new()));
        self.tapes
            .get_mut(&key)
            .and_then(|t| t.downcast_mut::<TapeArena<B, D>>())
            .expect("autograd tape type mismatch")
    }
}

fn with_tape_mut<B, D, R>(f: impl FnOnce(&mut TapeArena<B, D>) -> R) -> R
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let runtime = rt.get_or_insert_with(Runtime::new);
        let tape = runtime.tape_mut::<B, D>();
        f(tape)
    })
}

pub(crate) fn record_node<B, D>(
    output: TensorId,
    inputs: Vec<(TensorId, bool)>,
    op: Box<dyn BackwardOp<B, D>>,
    saved_tensors: Vec<Tensor<B, D>>,
) where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    let inputs: Vec<Input> = inputs
        .into_iter()
        .map(|(id, requires_grad)| Input { id, requires_grad })
        .collect();
    let backward = BackwardEntry { op, saved_tensors };
    with_tape_mut::<B, D, _>(|tape| tape.record(output, inputs, backward));
}

fn take_tape_nodes<B, D>() -> Vec<Node<B, D>>
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    with_tape_mut::<B, D, _>(TapeArena::take_nodes)
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

    fn saved(&self, idx: usize) -> &Tensor<B, D> {
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

pub fn backward<B, D>(loss: &Tensor<B, D>) -> Result<Grads<B, D>>
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

    let nodes = take_tape_nodes::<B, D>();
    let seed = if loss.numel() == 1 {
        let backend = loss.backend();
        Tensor::full(&backend, &[], D::one())?
    } else {
        Tensor::ones_like(loss)?
    };

    let loss_id = loss.tensor_id();
    let loss_tracked = nodes.iter().any(|n| n.output == loss_id);
    if !loss_tracked {
        let mut leaf = HashMap::new();
        leaf.insert(loss_id, seed);
        return Ok(Grads { leaf });
    }

    if nodes.is_empty() {
        return Err(Error::OpError(
            "cannot compute backward: loss is tracked but tape is empty".into(),
        ));
    }

    let mut grads: HashMap<TensorId, Tensor<B, D>> = HashMap::new();
    grads.insert(loss_id, seed);

    let produced: HashSet<TensorId> = nodes.iter().map(|n| n.output).collect();
    let mut leaf_required: HashSet<TensorId> = HashSet::new();
    for n in &nodes {
        for inp in &n.inputs {
            if inp.requires_grad && !produced.contains(&inp.id) {
                leaf_required.insert(inp.id);
            }
        }
    }

    for node in nodes.into_iter().rev() {
        let Some(grad_output) = grads.remove(&node.output) else {
            continue;
        };

        let ctx = BackwardContext::new(&node.backward.saved_tensors);
        let input_grads = node.backward.op.backward(&grad_output, &ctx)?;

        if input_grads.len() != node.inputs.len() {
            return Err(Error::OpError(format!(
                "backward op {} returned {} grads for {} inputs",
                node.backward.op.name(),
                input_grads.len(),
                node.inputs.len()
            )));
        }

        for (inp, gopt) in node.inputs.into_iter().zip(input_grads.into_iter()) {
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
            let acc = existing.add(&grad)?;
            map.insert(key, acc);
        }
    }
    Ok(())
}
