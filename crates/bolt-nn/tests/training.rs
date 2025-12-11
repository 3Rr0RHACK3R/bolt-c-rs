use std::sync::Arc;

use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::linear;
use bolt_nn::{Context, Eval, Grad, Model};

type B = CpuBackend;
type D = f32;

#[test]
fn test_grad_context_creation() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    // Can access autodiff backend
    let _ = ctx.autodiff();
}

#[test]
fn test_linear_forward_with_grad() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);
    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let spec = linear(4, 2);
    let layer = spec.build(&backend).unwrap();

    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

    // Forward pass with gradient context - params are wrapped as autodiff tensors
    let output = layer.forward(&grad_ctx, grad_ctx.input(&input)).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_backward_with_context() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);
    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    // Build a linear layer
    let mut layer = linear(2, 1).build(&backend).unwrap();

    // Input tensor
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();

    // Forward pass with gradient context
    let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();

    // Compute a simple loss (sum of output)
    let loss = output.sum(None, false).unwrap();

    // Backward pass - populates gradients on params
    let mut weight = &mut layer.weight;
    let mut bias = layer.bias.as_mut().unwrap();
    grad_ctx.backward(&loss, &mut [&mut weight, &mut bias]).unwrap();

    // Verify gradients were populated
    assert!(layer.weight.grad().is_some());
    assert!(layer.bias.as_ref().unwrap().grad().is_some());
}

#[test]
fn test_parameter_freeze() {
    use bolt_nn::layers::Linear;
    
    let backend = Arc::new(CpuBackend::new());

    let mut layer: Linear<B, D> = linear(4, 2).build(&backend).unwrap();

    // Params require grad by default
    assert!(layer.weight.requires_grad());
    assert!(layer.bias.as_ref().unwrap().requires_grad());

    // Freeze weight
    layer.weight.freeze();
    assert!(!layer.weight.requires_grad());

    // Unfreeze
    layer.weight.unfreeze();
    assert!(layer.weight.requires_grad());
}

#[test]
fn test_zero_grad() {
    use bolt_nn::layers::Linear;
    
    let backend = Arc::new(CpuBackend::new());

    let mut layer: Linear<B, D> = linear(4, 2).build(&backend).unwrap();

    // No gradients initially
    assert!(layer.weight.grad().is_none());

    // Zero grad should be no-op when no grads
    layer.weight.zero_grad();
    assert!(layer.weight.grad().is_none());
}
