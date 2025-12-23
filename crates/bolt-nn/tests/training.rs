use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_nn::layers::Linear;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn backward_populates_parameter_gradients() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);
    let layer = Linear::init(&store.sub("linear"), 2, 1, true).unwrap();

    store.zero_grad();

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = layer.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();

    store.backward(&loss).unwrap();

    assert!(layer.weight.grad().is_some());
    assert!(layer.bias.as_ref().unwrap().grad().is_some());
}

#[test]
fn freeze_unfreeze_toggles_requires_grad() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend, 0);
    let layer = Linear::init(&store.sub("linear"), 2, 1, true).unwrap();

    assert!(layer.weight.requires_grad());
    assert!(layer.bias.as_ref().unwrap().requires_grad());

    layer.weight.freeze();
    assert!(!layer.weight.requires_grad());

    layer.weight.unfreeze();
    assert!(layer.weight.requires_grad());
}

#[test]
fn zero_grad_clears_cached_gradients() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);
    let layer = Linear::init(&store.sub("linear"), 2, 1, true).unwrap();

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = layer.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    assert!(layer.weight.grad().is_some());
    store.zero_grad();
    assert!(layer.weight.grad().is_none());
}
