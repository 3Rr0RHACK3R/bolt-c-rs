use std::sync::Arc;

use bolt_core::backend::{AddOp, Backend, CopyOp, FillOp, SumOp};
use bolt_core::{BaseBackend, Float, Tensor};

use crate::Handle;
use crate::backward::BackwardContext;
use crate::error::{Error, Result};
use crate::gradients::{Gradients, insert_or_accumulate};
use crate::operations::Autodiff;
use crate::storage::AutodiffStorage;
use crate::utils::create_backward_seed;

pub trait AutodiffBackend<D: Float>: Backend {
    type InnerBackend: BaseBackend;

    fn autodiff(&self) -> &Autodiff<Self::InnerBackend, D>;
    fn inner_backend(&self) -> &Arc<Self::InnerBackend>;
}

impl<B, D> AutodiffBackend<D> for Autodiff<B, D>
where
    B: BaseBackend,
    D: Float,
{
    type InnerBackend = B;

    fn autodiff(&self) -> &Autodiff<B, D> {
        self
    }

    fn inner_backend(&self) -> &Arc<B> {
        &self.inner
    }
}

pub trait AutodiffTensorExt<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn requires_grad(self) -> Tensor<Autodiff<B, D>, D>;
    fn detach(&self) -> Tensor<B, D>;
    fn is_tracked(&self) -> bool;
    fn backward(&self) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>;
}

impl<B, D> AutodiffTensorExt<B, D> for Tensor<Autodiff<B, D>, D>
where
    B: BaseBackend,
    D: Float,
{
    fn requires_grad(self) -> Tensor<Autodiff<B, D>, D> {
        let backend = self.backend();
        let layout = self.layout().clone();
        let storage = self.storage();

        let new_storage = if backend.has_active_graph() && backend.is_grad_enabled() {
            let handle = backend
                .with_graph(|graph| {
                    graph.create_node(true, true, tinyvec::ArrayVec::new(), None, vec![])
                })
                .unwrap_or(Handle::NONE);

            AutodiffStorage::new(storage.inner.clone(), handle, true)
        } else {
            AutodiffStorage::new(storage.inner.clone(), Handle::NONE, true)
        };

        Tensor::from_parts(backend, new_storage, layout)
    }

    fn detach(&self) -> Tensor<B, D> {
        let backend = self.backend();
        Tensor::from_parts(
            backend.inner().clone(),
            self.storage().inner.clone(),
            self.layout().clone(),
        )
    }

    fn is_tracked(&self) -> bool {
        self.storage().requires_grad
    }

    fn backward(&self) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    {
        let backend = self.backend();

        if !backend.has_active_graph() {
            return Err(Error::NoActiveGraph);
        }

        let loss_handle = self.storage().handle();
        let graph_ref = backend.graph().read().unwrap();
        let graph = graph_ref.as_ref().ok_or(Error::NoActiveGraph)?;

        if !backend.is_grad_enabled() {
            return Err(Error::GradDisabled);
        }

        let loss_node = graph.get_node(loss_handle)?;
        if !loss_node.requires_grad {
            return Err(Error::LossNoGrad);
        }

        let inner_backend = backend.inner().clone();

        let seed = create_backward_seed(&inner_backend, self)?;

        let mut grad_map: std::collections::HashMap<Handle, Tensor<B, D>> =
            std::collections::HashMap::new();
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

            let ctx = BackwardContext::new(&backward_entry.saved_tensors, &inner_backend);
            let input_grads = backward_entry.op.backward(&grad_output, &ctx)?;

            for (input_handle, input_grad) in node.inputs.iter().zip(input_grads.into_iter()) {
                if input_handle.is_none() {
                    continue;
                }
                if let Some(grad) = input_grad {
                    insert_or_accumulate(&mut grad_map, *input_handle, grad)?;
                }
            }
        }

        let generation = graph.generation();
        let leaf_grads: std::collections::HashMap<Handle, Tensor<B, D>> = grad_map
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
