use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use bolt_core::backend::{AddOp, CopyOp, FillOp, SumOp};
use bolt_core::{Backend, OneValue, Tensor};

use crate::backward::BackwardContext;
use crate::error::{Error, Result};
use crate::gradients::{Gradients, insert_or_accumulate};
use crate::operations::Autodiff;
use crate::utils::create_backward_seed;
use crate::{Float, Handle};

pub struct GradContext<B, D>
where
    B: Backend,
    D: Float,
{
    autodiff: Autodiff<B, D>,
}

impl<B, D> GradContext<B, D>
where
    B: Backend,
    D: Float,
{
    pub(crate) fn new(autodiff: Autodiff<B, D>) -> Self {
        Self { autodiff }
    }

    pub fn autodiff(&self) -> &Autodiff<B, D> {
        &self.autodiff
    }

    pub fn backward(&self, loss: &Tensor<Autodiff<B, D>, D>) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
        D: OneValue,
    {
        let loss_handle = loss.storage().handle();
        self.backward_from_handle(loss_handle, loss)
    }

    fn backward_from_handle(
        &self,
        loss_handle: Handle,
        loss_tensor: &Tensor<Autodiff<B, D>, D>,
    ) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
        D: OneValue,
    {
        let graph_ref = self.autodiff.graph().read().unwrap();
        let graph = graph_ref.as_ref().ok_or(Error::NoActiveGraph)?;

        if !self.autodiff.is_grad_enabled() {
            return Err(Error::GradDisabled);
        }

        let loss_node = graph.get_node(loss_handle)?;
        if !loss_node.requires_grad {
            return Err(Error::LossNoGrad);
        }

        let inner_backend = self.autodiff.inner();

        let seed = create_backward_seed(inner_backend, loss_tensor)?;

        let mut grad_map: HashMap<Handle, Tensor<B, D>> = HashMap::new();
        grad_map.insert(loss_handle, seed);

        for node in graph.nodes_iter().rev() {
            let grad_output = match grad_map.get(&node.handle) {
                Some(g) => g.clone(),
                None => continue,
            };

            let backward_entry = match &node.backward_op {
                Some(entry) => entry,
                None => continue,
            };

            let ctx = BackwardContext::new(&backward_entry.saved_tensors, inner_backend);
            let input_grads = backward_entry.op.backward(&grad_output, &ctx)?;

            for (input_handle, input_grad) in node.inputs.iter().zip(input_grads.into_iter()) {
                if let Some(grad) = input_grad {
                    insert_or_accumulate(&mut grad_map, *input_handle, grad)?;
                }
            }
        }

        let generation = graph.generation();
        let leaf_grads: HashMap<Handle, Tensor<B, D>> = grad_map
            .into_iter()
            .filter(|(handle, _)| {
                graph
                    .get_node(*handle)
                    .map(|n| n.is_leaf && n.requires_grad)
                    .unwrap_or(false)
            })
            .collect();

        Ok(Gradients::new(leaf_grads, generation))
    }
}

impl<B, D> Drop for GradContext<B, D>
where
    B: Backend,
    D: Float,
{
    fn drop(&mut self) {
        *self.autodiff.graph().write().unwrap() = None;
    }
}

pub struct NoGradGuard<B, D>
where
    B: Backend,
    D: Float,
{
    autodiff_grad_enabled: Arc<RwLock<bool>>,
    prev: bool,
    _marker: PhantomData<(B, D)>,
}

impl<B, D> NoGradGuard<B, D>
where
    B: Backend,
    D: Float,
{
    pub fn new(autodiff: &Autodiff<B, D>) -> Self {
        let lock = autodiff.grad_enabled_lock().clone();
        let prev = *lock.read().unwrap();
        *lock.write().unwrap() = false;
        Self {
            autodiff_grad_enabled: lock,
            prev,
            _marker: PhantomData,
        }
    }
}

impl<B, D> Drop for NoGradGuard<B, D>
where
    B: Backend,
    D: Float,
{
    fn drop(&mut self) {
        *self.autodiff_grad_enabled.write().unwrap() = self.prev;
    }
}
