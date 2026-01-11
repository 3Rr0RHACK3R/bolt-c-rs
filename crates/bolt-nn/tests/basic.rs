use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{Flatten, Linear, Relu, Sigmoid, Tanh};
use bolt_nn::{ForwardCtx, Init, Module, Store};
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
    assert_eq!(output.shape().as_slice(), &[1, 2]);
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
    // Order is determined by ParamId (insertion order)
    assert_eq!(
        keys,
        vec!["linear.weight".to_string(), "linear.bias".to_string()]
    );

    let params = store.trainable();
    assert_eq!(params.len(), 2);
    // Order matches the keys order
    assert_eq!(params[0].shape().as_slice(), &[3, 2]);
    assert_eq!(params[1].shape().as_slice(), &[3]);
}

#[test]
fn flatten_forward_reshapes_4d_to_2d() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 3, 4, 5]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape().as_slice(), &[2, 60]);
}

#[test]
fn flatten_forward_preserves_2d_tensor() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[4, 8]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape().as_slice(), &[4, 8]);
}

#[test]
fn flatten_forward_handles_single_sample() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 16]).unwrap();

    let layer = Flatten::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape().as_slice(), &[1, 16]);
}

#[test]
fn sigmoid_forward_produces_values_in_zero_one_range() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[-10.0, 0.0, 10.0], &[3]).unwrap();

    let layer = Sigmoid::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    for val in data.iter() {
        assert!(*val > 0.0 && *val < 1.0);
    }
    assert!(data[1] > data[0]);
    assert!(data[2] > data[1]);
}

#[test]
fn tanh_forward_produces_values_in_negative_one_one_range() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[-10.0, 0.0, 10.0], &[3]).unwrap();

    let layer = Tanh::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    for val in data.iter() {
        assert!(*val >= -1.0 && *val <= 1.0);
    }
    assert!((data[0] - (-1.0)).abs() < 0.01); // tanh(-10) ≈ -1
    assert!((data[1] - 0.0).abs() < 1e-6); // tanh(0) = 0
    assert!((data[2] - 1.0).abs() < 0.01); // tanh(10) ≈ 1
}

#[test]
fn kaiming_normal_initialization() {

    let _backend = Arc::new(CpuBackend::new());
    let shape = &[64, 32];

    let key = bolt_rng::RngKey::from_seed(42);
    let weights = bolt_nn::fill::<f32>(shape, Init::KaimingNormal { a: 0.0 }, key).unwrap();

    assert_eq!(weights.len(), 64 * 32);

    let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
    let variance: f32 =
        weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / weights.len() as f32;
    let std = variance.sqrt();

    let expected_std = (2.0f64 / 32.0).sqrt() as f32;
    assert!((std - expected_std).abs() < 0.1);
}

#[test]
fn store_new_with_rng_works() {
    let backend = Arc::new(CpuBackend::new());
    let init_key = bolt_rng::RngKey::from_seed(42).derive("init");
    let store = Store::<B, D>::new_with_init_key(backend.clone(), init_key);
    let layer = Linear::init(&store.sub("linear"), 4, 2, true).unwrap();

    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape().as_slice(), &[1, 2]);
}

#[test]
fn store_new_with_rng_deterministic() {
    let backend = Arc::new(CpuBackend::new());

    // Create two stores with same init key
    let init_key1 = bolt_rng::RngKey::from_seed(123).derive("init");
    let init_key2 = bolt_rng::RngKey::from_seed(123).derive("init");

    let store1 = Store::<B, D>::new_with_init_key(backend.clone(), init_key1);
    let store2 = Store::<B, D>::new_with_init_key(backend.clone(), init_key2);

    let layer1 = Linear::init(&store1.sub("linear"), 2, 1, true).unwrap();
    let layer2 = Linear::init(&store2.sub("linear"), 2, 1, true).unwrap();

    // Same RNG should produce same initial weights
    assert_eq!(
        layer1.weight.tensor().to_vec().unwrap(),
        layer2.weight.tensor().to_vec().unwrap()
    );
}

#[test]
fn param_count_counts_all_trainable_parameters() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend, 0);
    
    // Create a linear layer: weight [3, 2] = 6 params, bias [3] = 3 params
    let _layer1 = Linear::init(&store.sub("linear1"), 2, 3, true).unwrap();
    
    // Create another linear layer: weight [4, 3] = 12 params, bias [4] = 4 params
    let _layer2 = Linear::init(&store.sub("linear2"), 3, 4, true).unwrap();
    
    // Total: 6 + 3 + 12 + 4 = 25 params
    assert_eq!(store.param_count(), 25);
}

#[test]
fn param_count_handles_empty_store() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend, 0);
    
    assert_eq!(store.param_count(), 0);
}
