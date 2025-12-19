use bolt_autodiff::Float;
use bolt_autodiff::HasParams;
use bolt_autodiff::Parameter;
use bolt_core::BaseBackend;
use bolt_core::Tensor;
use bolt_core::backend::Backend;
use bolt_core::backend::ReshapeOp;

use crate::context::Context;
use crate::error::Result;
use crate::mode::Mode;
use crate::model::Model;

pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: usize) -> Self {
        Self { start_dim }
    }
}

impl<B, D> HasParams<B, D> for Flatten
where
    B: BaseBackend,
    D: Float,
{
    fn visit_params<'a>(&'a self, _f: &mut dyn FnMut(&'a Parameter<B, D>)) {}
    fn visit_params_mut<'a>(&'a mut self, _f: &mut dyn FnMut(&'a mut Parameter<B, D>)) {}
    fn param_count(&self) -> usize {
        0
    }
}

impl<B, D, M> Model<B, D, M> for Flatten
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
    M::Backend: Backend + ReshapeOp<D>,
{
    type Input = Tensor<M::Backend, D>;
    type Output = Result<Tensor<M::Backend, D>>;

    fn forward(&self, _ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output {
        let shape = input.shape();
        if self.start_dim >= shape.len() {
            return Ok(input);
        }
        let flat_dims: usize = shape[self.start_dim..].iter().product();
        let new_shape = if self.start_dim == 0 {
            vec![flat_dims]
        } else {
            let mut s = shape[..self.start_dim].to_vec();
            s.push(flat_dims);
            s
        };
        Ok(input.reshape(&new_shape)?)
    }
}

pub fn flatten(start_dim: usize) -> Flatten {
    Flatten::new(start_dim)
}
