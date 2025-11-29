use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::{Float, Graph, Handle, error::Result, graph::MAX_SHAPE_RANK};

#[derive(Clone, Copy)]
pub struct GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    graph: &'g Graph<B, D>,
    handle: Handle,
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub(crate) fn new(graph: &'g Graph<B, D>, handle: Handle) -> Self {
        Self { graph, handle }
    }

    pub fn tensor(&self) -> Result<Tensor<B, D>> {
        self.graph.get_tensor(self.handle)
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    pub fn shape(&self) -> Result<ArrayVec<[usize; MAX_SHAPE_RANK]>> {
        self.graph.get_node_shape(self.handle)
    }

    pub fn requires_grad(&self) -> Result<bool> {
        self.graph.get_node_requires_grad(self.handle)
    }

    pub fn is_leaf(&self) -> Result<bool> {
        self.graph.get_node_is_leaf(self.handle)
    }

    pub fn detach(&self) -> Result<GradTensor<'g, B, D>> {
        let tensor = self.tensor()?;
        Ok(self.graph.input(&tensor))
    }

    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        self.graph
            .set_node_requires_grad(self.handle, requires_grad)
    }

    pub fn graph(&self) -> &'g Graph<B, D> {
        self.graph
    }
}

pub trait TensorLike<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D>;
}

pub trait Attach<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn attach(self, graph: &'g Graph<B, D>) -> AttachBuilder<'g, B, D>;
}

pub struct AttachBuilder<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    graph: &'g Graph<B, D>,
    tensor: Tensor<B, D>,
}

impl<'g, B, D> AttachBuilder<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub fn with_grad(self) -> GradTensor<'g, B, D> {
        self.graph.param(&self.tensor)
    }

    pub fn no_grad(self) -> GradTensor<'g, B, D> {
        self.graph.input(&self.tensor)
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, _graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        self
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for &GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, _graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        self.clone()
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for Tensor<B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        graph.input(&self)
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for &Tensor<B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        graph.input(self)
    }
}

impl<'g, B, D> Attach<'g, B, D> for Tensor<B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn attach(self, graph: &'g Graph<B, D>) -> AttachBuilder<'g, B, D> {
        AttachBuilder {
            graph,
            tensor: self,
        }
    }
}
