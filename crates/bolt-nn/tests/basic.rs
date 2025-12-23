use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{Flatten, Linear, Relu};
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn linear_forward_produces_expected_shape() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);
    let layer = Linear::init(&store.sub("linear"), 4, 2, true).unwrap();

    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn relu_forward_clamps_negative_values() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();

    let layer = Relu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();
    assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn store_registers_linear_params_with_expected_keys() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend, 0);
    let _ = Linear::init(&store.sub("linear"), 2, 3, true).unwrap();

    let keys: Vec<String> = store
        .named_trainable()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(
        keys,
        vec!["linear.bias".to_string(), "linear.weight".to_string()]
    );

    let params = store.trainable();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].shape(), &[3]);
    assert_eq!(params[1].shape(), &[3, 2]);
}

#[test]
fn flatten_forward_reshapes_4d_to_2d() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 3, 4, 5]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[2, 60]);
}

#[test]
fn flatten_forward_preserves_2d_tensor() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[4, 8]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[4, 8]);
}

#[test]
fn flatten_forward_handles_single_sample() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 16]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 16]);
}
