use bolt_autodiff::{Autodiff, Float, HasParams, Parameter};
use bolt_core::Tensor;

use crate::compute::Compute;
use crate::context::Context;
use crate::error::Result;
use crate::mode::{Eval, Grad};
use crate::model::Model;
use crate::run_mode::{RunMode, Trainable};
use crate::visit::NamedParams;

pub trait Layer<B, D>: HasParams<B, D> + Trainable + Send + Sync
where
    B: Compute<D>,
    D: Float,
{
    fn forward_eval(
        &self,
        ctx: &Context<B, D, Eval<B, D>>,
        input: Tensor<B, D>,
    ) -> Result<Tensor<B, D>>;

    fn forward_grad(
        &self,
        ctx: &Context<B, D, Grad<B, D>>,
        input: Tensor<Autodiff<B, D>, D>,
    ) -> Result<Tensor<Autodiff<B, D>, D>>;
}

impl<T, B, D> Layer<B, D> for T
where
    B: Compute<D>,
    D: Float,
    T: HasParams<B, D>
        + Trainable
        + Model<B, D, Eval<B, D>, Input = Tensor<B, D>, Output = Result<Tensor<B, D>>>
        + Model<
            B,
            D,
            Grad<B, D>,
            Input = Tensor<Autodiff<B, D>, D>,
            Output = Result<Tensor<Autodiff<B, D>, D>>,
        >
        + Send
        + Sync,
{
    fn forward_eval(
        &self,
        ctx: &Context<B, D, Eval<B, D>>,
        input: Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        <T as Model<B, D, Eval<B, D>>>::forward(self, ctx, input)
    }

    fn forward_grad(
        &self,
        ctx: &Context<B, D, Grad<B, D>>,
        input: Tensor<Autodiff<B, D>, D>,
    ) -> Result<Tensor<Autodiff<B, D>, D>> {
        <T as Model<B, D, Grad<B, D>>>::forward(self, ctx, input)
    }
}

pub struct Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    layers: Vec<LayerEntry<B, D>>,
    run_mode: RunMode,
}

struct LayerEntry<B, D>
where
    B: Compute<D>,
    D: Float,
{
    name: String,
    layer: Box<dyn Layer<B, D> + Send + Sync>,
}

impl<B, D> Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            run_mode: RunMode::Train,
        }
    }

    pub fn push<L>(mut self, layer: L) -> Self
    where
        L: Layer<B, D> + Send + Sync + 'static,
    {
        let name = self.layers.len().to_string();
        self.layers.push(LayerEntry {
            name,
            layer: Box::new(layer),
        });
        self
    }

    pub fn push_named<L>(mut self, name: impl Into<String>, layer: L) -> Self
    where
        L: Layer<B, D> + Send + Sync + 'static,
    {
        self.layers.push(LayerEntry {
            name: name.into(),
            layer: Box::new(layer),
        });
        self
    }
}

impl<B, D> Default for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> Model<B, D, Eval<B, D>> for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    type Input = Tensor<B, D>;
    type Output = Result<Tensor<B, D>>;

    fn forward(&self, ctx: &Context<B, D, Eval<B, D>>, input: Self::Input) -> Self::Output {
        let mut x = input;
        for layer in &self.layers {
            x = layer.layer.forward_eval(ctx, x)?;
        }
        Ok(x)
    }
}

impl<B, D> Model<B, D, Grad<B, D>> for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    type Input = Tensor<Autodiff<B, D>, D>;
    type Output = Result<Tensor<Autodiff<B, D>, D>>;

    fn forward(&self, ctx: &Context<B, D, Grad<B, D>>, input: Self::Input) -> Self::Output {
        let mut x = input;
        for layer in &self.layers {
            x = layer.layer.forward_grad(ctx, x)?;
        }
        Ok(x)
    }
}

impl<B, D> HasParams<B, D> for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    fn visit_params<'a>(&'a self, f: &mut dyn FnMut(&'a Parameter<B, D>)) {
        for layer in &self.layers {
            layer.layer.visit_params(f);
        }
    }

    fn visit_params_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Parameter<B, D>)) {
        for layer in &mut self.layers {
            layer.layer.visit_params_mut(f);
        }
    }

    fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.layer.param_count()).sum()
    }
}

impl<B, D> NamedParams<B, D> for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    fn visit_named_params(&self, f: &mut dyn FnMut(&str, &Parameter<B, D>)) {
        for entry in &self.layers {
            let layer_prefix = format!("layers.{}", entry.name);
            let mut i = 0usize;
            entry.layer.visit_params(&mut |p| {
                let key = format!("{}.p{}.{}", layer_prefix, i, p.display_name());
                i += 1;
                f(&key, p);
            });
        }
    }

    fn visit_named_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B, D>)) {
        for entry in &mut self.layers {
            let layer_prefix = format!("layers.{}", entry.name);
            let mut i = 0usize;
            entry.layer.visit_params_mut(&mut |p| {
                let key = format!("{}.p{}.{}", layer_prefix, i, p.display_name());
                i += 1;
                f(&key, p);
            });
        }
    }
}

impl<B, D> Trainable for Seq<B, D>
where
    B: Compute<D>,
    D: Float,
{
    fn set_run_mode(&mut self, mode: RunMode) {
        self.run_mode = mode;
        for entry in &mut self.layers {
            entry.layer.set_run_mode(mode);
        }
    }
}
