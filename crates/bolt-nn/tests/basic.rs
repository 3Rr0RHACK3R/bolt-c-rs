use std::sync::Arc;

use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::{HasParams, Linear, linear};
use bolt_nn::{Context, Eval, Model};

type B = CpuBackend;
type D = f32;
type M = Eval<B, D>;

#[test]
fn test_linear_forward() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B, D, M>::eval(&backend);

    let spec = linear(4, 2);
    let layer: Linear<B, D> = spec.build(&backend).unwrap();

    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let output = layer.forward(&ctx, ctx.input(&input)).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_relu_forward() {
    use bolt_nn::layers::relu;

    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B, D, M>::eval(&backend);

    let input = Tensor::<B, D>::from_slice(&backend, &[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
    let layer = relu();
    let output = layer.forward(&ctx, ctx.input(&input)).unwrap();

    let data = output.to_vec().unwrap();
    assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_context_backend_access() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B, D, M>::eval(&backend);

    let _ = ctx.backend();
    let _ = ctx.device();
}

#[test]
fn test_linear_params_access() {
    let backend = Arc::new(CpuBackend::new());

    let layer: Linear<B, D> = linear(4, 2).build(&backend).unwrap();

    let params = layer.params();
    assert_eq!(params.len(), 2);

    assert_eq!(params[0].tensor().shape(), &[2, 4]);
    assert_eq!(params[1].tensor().shape(), &[2]);
}
