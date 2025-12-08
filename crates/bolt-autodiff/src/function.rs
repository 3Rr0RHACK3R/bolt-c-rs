use std::marker::PhantomData;
use std::sync::Arc;

use bolt_core::backend::{AddOp, Backend, CopyOp, FillOp};
use bolt_core::Tensor;
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::operations::Autodiff;
use crate::storage::AutodiffStorage;
use crate::tensor_ext::AutodiffTensorExt;
use crate::utils::create_saved_tensor;
use crate::{Float, Handle};

pub trait Function<B, D>: Send + Sync + 'static
where
    B: Backend,
    D: Float,
{
    type Ctx: Send + Sync + Default + Clone;

    fn forward(ctx: &mut Self::Ctx, inputs: &[&Tensor<B, D>]) -> Result<Vec<Tensor<B, D>>>;

    fn backward(
        ctx: &Self::Ctx,
        grad_outputs: &[Option<&Tensor<B, D>>],
    ) -> Result<Vec<Option<Tensor<B, D>>>>;

    fn apply(inputs: &[&Tensor<Autodiff<B, D>, D>]) -> Result<Vec<Tensor<Autodiff<B, D>, D>>>
    where
        Self: Sized,
        B: AddOp<D> + CopyOp<D> + FillOp<D>,
    {
        apply_fn::<Self, B, D>(inputs)
    }
}

fn apply_fn<F, B, D>(inputs: &[&Tensor<Autodiff<B, D>, D>]) -> Result<Vec<Tensor<Autodiff<B, D>, D>>>
where
    F: Function<B, D>,
    B: Backend + AddOp<D> + CopyOp<D> + FillOp<D>,
    D: Float,
{
    if inputs.is_empty() {
        return Err(crate::error::Error::EmptyInputs);
    }

    let autodiff = inputs[0].backend();
    let inner_backend = autodiff.inner();

    let inner_inputs: Vec<Tensor<B, D>> = inputs.iter().map(|t| t.detach()).collect();
    let inner_refs: Vec<&Tensor<B, D>> = inner_inputs.iter().collect();

    let mut ctx = F::Ctx::default();
    let outputs = F::forward(&mut ctx, &inner_refs)?;

    if outputs.is_empty() {
        return Err(crate::error::Error::EmptyOutputs);
    }

    let needs_grad = autodiff.is_grad_enabled()
        && autodiff.has_active_graph()
        && inputs.iter().any(|t| t.is_tracked());

    if !needs_grad {
        return outputs
            .into_iter()
            .map(|t| {
                let storage = AutodiffStorage::new(t.storage().clone(), Handle::NONE, false);
                Ok(Tensor::from_parts(autodiff.clone(), storage, t.layout().clone()))
            })
            .collect();
    }

    let input_handles: ArrayVec<[Handle; MAX_INPUTS]> = inputs
        .iter()
        .take(MAX_INPUTS)
        .map(|t| t.storage().handle())
        .collect();

    let saved_inputs: Vec<Tensor<B, D>> = inputs
        .iter()
        .map(|t| create_saved_tensor(inner_backend, t.storage().inner(), t.layout()))
        .collect();

    let num_outputs = outputs.len();
    let shared_ctx = Arc::new(ctx);
    
    let mut result = Vec::with_capacity(num_outputs);
    for (idx, output) in outputs.into_iter().enumerate() {
        let layout = output.layout().clone();
        let inner_storage = output.storage().clone();

        let storage = if idx == 0 {
            let backward_op = FunctionBackward::<F, B, D> {
                ctx: shared_ctx.clone(),
                num_outputs,
                _marker: PhantomData,
            };
            let inner_storage_clone = inner_storage.clone();
            autodiff
                .with_graph(|graph| {
                    let handle = graph.create_node(
                        true,
                        false,
                        input_handles.clone(),
                        Some(Box::new(backward_op)),
                        saved_inputs.clone(),
                    );
                    AutodiffStorage::new(inner_storage_clone, handle, true)
                })
                .unwrap_or_else(|| AutodiffStorage::new(inner_storage, Handle::NONE, true))
        } else {
            let inner_storage_clone = inner_storage.clone();
            autodiff
                .with_graph(|graph| {
                    let handle = graph.create_node(true, false, ArrayVec::new(), None, vec![]);
                    AutodiffStorage::new(inner_storage_clone, handle, true)
                })
                .unwrap_or_else(|| AutodiffStorage::new(inner_storage, Handle::NONE, true))
        };

        result.push(Tensor::from_parts(autodiff.clone(), storage, layout));
    }

    Ok(result)
}

struct FunctionBackward<F, B, D>
where
    F: Function<B, D>,
    B: Backend,
    D: Float,
{
    ctx: Arc<F::Ctx>,
    num_outputs: usize,
    _marker: PhantomData<fn() -> (B, D)>,
}

impl<F, B, D> BackwardOp<B, D> for FunctionBackward<F, B, D>
where
    F: Function<B, D>,
    B: Backend + AddOp<D> + CopyOp<D> + FillOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let mut grad_outputs: Vec<Option<&Tensor<B, D>>> = Vec::with_capacity(self.num_outputs);
        grad_outputs.push(Some(grad_output));
        for _ in 1..self.num_outputs {
            grad_outputs.push(None);
        }

        let input_grads = F::backward(&self.ctx, &grad_outputs)?;

        let mut result = ArrayVec::new();
        for grad in input_grads.into_iter().take(MAX_INPUTS) {
            result.push(grad);
        }
        Ok(result)
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<F>()
    }
}
