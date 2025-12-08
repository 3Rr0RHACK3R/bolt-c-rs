use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::Handle;
use crate::backward::{BackwardOp, MAX_INPUTS};
use crate::error::{Error, Result};

pub(crate) struct BackwardEntry<B, D>
where
    B: Backend,
    D: Float,
{
    pub op: Box<dyn BackwardOp<B, D>>,
    pub saved_tensors: Vec<Tensor<B, D>>,
}

pub(crate) struct Node<B, D>
where
    B: Backend,
    D: Float,
{
    pub handle: Handle,
    pub inputs: ArrayVec<[Handle; MAX_INPUTS]>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub backward_op: Option<BackwardEntry<B, D>>,
}

pub struct Graph<B, D>
where
    B: Backend,
    D: Float,
{
    nodes: Vec<Node<B, D>>,
    generation: u32,
}

impl<B, D> Default for Graph<B, D>
where
    B: Backend,
    D: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> Graph<B, D>
where
    B: Backend,
    D: Float,
{
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            generation: 0,
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn create_node(
        &mut self,
        requires_grad: bool,
        is_leaf: bool,
        inputs: ArrayVec<[Handle; MAX_INPUTS]>,
        backward_op: Option<Box<dyn BackwardOp<B, D>>>,
        saved_tensors: Vec<Tensor<B, D>>,
    ) -> Handle {
        let index: u32 = self
            .nodes
            .len()
            .try_into()
            .expect("graph node count exceeded u32::MAX");
        let handle = Handle::new(index, self.generation);

        let backward_entry = backward_op.map(|op| BackwardEntry { op, saved_tensors });

        self.nodes.push(Node {
            handle,
            inputs,
            requires_grad,
            is_leaf,
            backward_op: backward_entry,
        });

        handle
    }

    pub fn get_node(&self, handle: Handle) -> Result<&Node<B, D>> {
        if handle.generation != self.generation {
            return Err(Error::stale_handle());
        }
        self.nodes
            .get(handle.index as usize)
            .ok_or_else(|| Error::handle_out_of_bounds(handle.index, self.nodes.len()))
    }

    pub fn nodes_iter(&self) -> impl DoubleEndedIterator<Item = &Node<B, D>> {
        self.nodes.iter()
    }
}
