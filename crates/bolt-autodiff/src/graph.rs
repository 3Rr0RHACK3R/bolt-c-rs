use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use bolt_core::backend::{AddOp, FillOp};
use bolt_core::shape;
use bolt_core::{Backend, OneValue, Tensor};
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::{Error, Result};
use crate::gradients::insert_or_accumulate;
use crate::{Float, GradTensor, Gradients, Handle};

pub(crate) const MAX_SHAPE_RANK: usize = shape::MAX_RANK;

pub(crate) struct Node {
    pub handle: Handle,
    pub inputs: ArrayVec<[Handle; MAX_INPUTS]>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub backward_op_idx: Option<usize>,
    pub shape: ArrayVec<[usize; MAX_SHAPE_RANK]>,
}

pub(crate) struct BackwardEntry<B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub op: Box<dyn BackwardOp<B, D>>,
    pub saved_idx: usize,
}

pub struct Graph<B, D>
where
    B: Backend<D>,
    D: Float,
{
    backend: Arc<B>,
    nodes: RefCell<Vec<Node>>,
    tensors: RefCell<HashMap<Handle, Tensor<B, D>>>,
    backward_ops: RefCell<Vec<BackwardEntry<B, D>>>,
    saved_tensors: RefCell<Vec<Vec<Tensor<B, D>>>>,
    generation: Cell<u32>,
    grad_enabled: Cell<bool>,
}

impl<B, D> Graph<B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            nodes: RefCell::new(Vec::new()),
            tensors: RefCell::new(HashMap::new()),
            backward_ops: RefCell::new(Vec::new()),
            saved_tensors: RefCell::new(Vec::new()),
            generation: Cell::new(0),
            grad_enabled: Cell::new(true),
        }
    }

    pub fn variable<'g>(&'g self, tensor: &Tensor<B, D>) -> GradTensor<'g, B, D> {
        self.create_node(tensor.clone(), true, true, ArrayVec::new(), None)
    }

    pub fn constant<'g>(&'g self, tensor: &Tensor<B, D>) -> GradTensor<'g, B, D> {
        self.create_node(tensor.clone(), false, true, ArrayVec::new(), None)
    }

    pub fn clear(&self) {
        self.nodes.borrow_mut().clear();
        self.tensors.borrow_mut().clear();
        self.backward_ops.borrow_mut().clear();
        self.saved_tensors.borrow_mut().clear();
        self.generation.set(self.generation.get().wrapping_add(1));
    }

    pub fn no_grad(&self) -> NoGradGuard<'_, B, D> {
        let prev = self.grad_enabled.get();
        self.grad_enabled.set(false);
        NoGradGuard { graph: self, prev }
    }

    pub fn is_grad_enabled(&self) -> bool {
        self.grad_enabled.get()
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn generation(&self) -> u32 {
        self.generation.get()
    }

    pub fn backward<'g>(&'g self, loss: &GradTensor<'g, B, D>) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D>,
        D: OneValue,
    {
        self.backward_impl(loss.handle())
    }

    fn backward_impl(&self, loss_handle: Handle) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D>,
        D: OneValue,
    {
        self.validate_handle(loss_handle)?;

        if !self.grad_enabled.get() {
            return Err(Error::GradDisabled);
        }

        {
            let nodes = self.nodes.borrow();
            let loss_node = &nodes[loss_handle.index as usize];
            if !loss_node.requires_grad {
                return Err(Error::LossNoGrad);
            }
        }

        let loss_tensor = self.get_tensor(loss_handle)?;

        let seed = if loss_tensor.numel() == 1 {
            Tensor::full(&self.backend, &[], OneValue::one())?
        } else {
            Tensor::ones(&self.backend, loss_tensor.shape())?
        };

        let mut grad_map: HashMap<Handle, Tensor<B, D>> = HashMap::new();
        grad_map.insert(loss_handle, seed);

        let nodes = self.nodes.borrow();
        let backward_ops = self.backward_ops.borrow();
        let saved_tensors = self.saved_tensors.borrow();

        for node in nodes.iter().rev() {
            let grad_output = match grad_map.get(&node.handle) {
                Some(g) => g.clone(),
                None => continue,
            };

            let backward_op_idx = match node.backward_op_idx {
                Some(idx) => idx,
                None => continue,
            };

            let backward_entry =
                backward_ops
                    .get(backward_op_idx)
                    .ok_or(Error::BackwardOpNotFound {
                        idx: backward_op_idx,
                    })?;

            let saved =
                saved_tensors
                    .get(backward_entry.saved_idx)
                    .ok_or(Error::SavedTensorsNotFound {
                        idx: backward_entry.saved_idx,
                    })?;

            let ctx = BackwardContext::new(saved, &self.backend);
            let input_grads = backward_entry.op.backward(&grad_output, &ctx)?;

            for (input_handle, input_grad) in
                node.inputs.as_slice().iter().zip(input_grads.into_iter())
            {
                if let Some(grad) = input_grad {
                    insert_or_accumulate(&mut grad_map, *input_handle, grad)?;
                }
            }
        }

        let leaf_grads: HashMap<Handle, Tensor<B, D>> = grad_map
            .into_iter()
            .filter(|(handle, _)| {
                let node = &nodes[handle.index as usize];
                node.is_leaf && node.requires_grad
            })
            .collect();

        Ok(Gradients::new(leaf_grads, self.generation.get()))
    }

    pub(crate) fn validate_handle(&self, handle: Handle) -> Result<()> {
        if handle.generation != self.generation.get() {
            return Err(Error::stale_handle());
        }
        let nodes = self.nodes.borrow();
        if handle.index as usize >= nodes.len() {
            return Err(Error::handle_out_of_bounds(handle.index, nodes.len()));
        }
        Ok(())
    }

    pub(crate) fn get_tensor(&self, handle: Handle) -> Result<Tensor<B, D>> {
        self.validate_handle(handle)?;
        self.tensors
            .borrow()
            .get(&handle)
            .cloned()
            .ok_or(Error::TensorNotFound { handle })
    }

    pub(crate) fn get_node_shape(
        &self,
        handle: Handle,
    ) -> Result<ArrayVec<[usize; MAX_SHAPE_RANK]>> {
        self.validate_handle(handle)?;
        let nodes = self.nodes.borrow();
        Ok(nodes[handle.index as usize].shape.clone())
    }

    pub(crate) fn get_node_requires_grad(&self, handle: Handle) -> Result<bool> {
        self.validate_handle(handle)?;
        let nodes = self.nodes.borrow();
        Ok(nodes[handle.index as usize].requires_grad)
    }

    pub(crate) fn get_node_is_leaf(&self, handle: Handle) -> Result<bool> {
        self.validate_handle(handle)?;
        let nodes = self.nodes.borrow();
        Ok(nodes[handle.index as usize].is_leaf)
    }

    pub(crate) fn set_node_requires_grad(&self, handle: Handle, requires_grad: bool) -> Result<()> {
        self.validate_handle(handle)?;
        let mut nodes = self.nodes.borrow_mut();
        let node = &mut nodes[handle.index as usize];
        if !node.is_leaf {
            return Err(Error::NotALeaf {
                reason: "set_requires_grad only valid for leaf tensors".into(),
            });
        }
        node.requires_grad = requires_grad;
        Ok(())
    }

    pub(crate) fn save_tensors_for_backward(&self, tensors: Vec<Tensor<B, D>>) -> usize {
        let mut saved = self.saved_tensors.borrow_mut();
        let idx = saved.len();
        saved.push(tensors);
        idx
    }

    fn register_backward_op(
        &self,
        backward_op: Option<(Box<dyn BackwardOp<B, D>>, usize)>,
    ) -> Option<usize> {
        backward_op.map(|(op, saved_idx)| {
            let mut ops = self.backward_ops.borrow_mut();
            let idx = ops.len();
            ops.push(BackwardEntry { op, saved_idx });
            idx
        })
    }

    fn collect_shape(tensor: &Tensor<B, D>) -> ArrayVec<[usize; MAX_SHAPE_RANK]> {
        let mut shape = ArrayVec::new();
        for &dim in tensor.shape() {
            shape.push(dim);
        }
        shape
    }

    pub(crate) fn create_node<'g>(
        &'g self,
        tensor: Tensor<B, D>,
        requires_grad: bool,
        is_leaf: bool,
        inputs: ArrayVec<[Handle; MAX_INPUTS]>,
        backward_op: Option<(Box<dyn BackwardOp<B, D>>, usize)>,
    ) -> GradTensor<'g, B, D> {
        let mut nodes = self.nodes.borrow_mut();
        let index: u32 = nodes
            .len()
            .try_into()
            .expect("graph node count exceeded u32::MAX");
        let generation = self.generation.get();
        let handle = Handle::new(index, generation);

        let backward_op_idx = self.register_backward_op(backward_op);
        let shape = Self::collect_shape(&tensor);

        nodes.push(Node {
            handle,
            inputs,
            requires_grad,
            is_leaf,
            backward_op_idx,
            shape,
        });

        self.tensors.borrow_mut().insert(handle, tensor);

        GradTensor::new(self, handle)
    }
}

pub struct NoGradGuard<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    graph: &'g Graph<B, D>,
    prev: bool,
}

impl<B, D> Drop for NoGradGuard<'_, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn drop(&mut self) {
        self.graph.grad_enabled.set(self.prev);
    }
}
