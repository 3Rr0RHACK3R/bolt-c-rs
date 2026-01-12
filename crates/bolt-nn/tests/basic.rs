use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{Flatten, Gelu, LeakyRelu, Linear, Relu, Sigmoid, Tanh};
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
fn leaky_relu_forward_scales_negatives() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

    let layer = LeakyRelu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    // Default slope is 0.01, so:
    // -2.0 * 0.01 = -0.02
    // -1.0 * 0.01 = -0.01
    // 0.0 remains 0.0
    // 1.0 remains 1.0
    // 2.0 remains 2.0
    let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];
    for (actual, exp) in data.iter().zip(expected.iter()) {
        assert!(
            (actual - exp).abs() < 1e-6,
            "Expected {}, got {}",
            exp,
            actual
        );
    }
}

#[test]
fn leaky_relu_forward_with_custom_slope() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

    let layer = LeakyRelu::with_negative_slope(0.1);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    // Slope is 0.1, so:
    // -2.0 * 0.1 = -0.2
    // -1.0 * 0.1 = -0.1
    let expected = vec![-0.2, -0.1, 0.0, 1.0, 2.0];
    for (actual, exp) in data.iter().zip(expected.iter()) {
        assert!(
            (actual - exp).abs() < 1e-6,
            "Expected {}, got {}",
            exp,
            actual
        );
    }
}

#[test]
fn leaky_relu_forward_clamps_nan() {
    let backend = Arc::new(CpuBackend::new());
    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, f32::NAN, -1.0], &[3]).unwrap();

    let layer = LeakyRelu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    // NaN should be clamped to 0.0 (consistent with relu behavior)
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 0.0).abs() < 1e-6);
    assert!((data[2] - (-0.01)).abs() < 1e-6);
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

#[test]
fn gelu_forward_produces_expected_outputs() {
    let backend = Arc::new(CpuBackend::new());
    // Test values at key points
    let input = Tensor::<B, D>::from_slice(&backend, &[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

    let layer = Gelu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    // GELU(0) = 0
    assert!(
        (data[2] - 0.0).abs() < 1e-5,
        "GELU(0) should be ~0, got {}",
        data[2]
    );

    // GELU(x) > 0 for x > 0
    assert!(data[3] > 0.0, "GELU(1) should be positive, got {}", data[3]);
    assert!(data[4] > 0.0, "GELU(2) should be positive, got {}", data[4]);

    // GELU(x) < 0 for small negative x, but approaches 0 asymptotically
    assert!(
        data[0] < 0.0 && data[0] > -0.1,
        "GELU(-2) should be small negative, got {}",
        data[0]
    );
    assert!(
        data[1] < 0.0 && data[1] > -0.2,
        "GELU(-1) should be small negative, got {}",
        data[1]
    );

    // GELU(1) ≈ 0.841 (from the formula)
    assert!(
        (data[3] - 0.841).abs() < 0.01,
        "GELU(1) should be ~0.841, got {}",
        data[3]
    );
}

#[test]
fn gelu_forward_matches_known_values() {
    let backend = Arc::new(CpuBackend::new());

    // Test that GELU approximately passes through these known points
    // Reference values from PyTorch/TensorFlow
    let test_cases: [(f32, f32); 5] = [
        (0.0, 0.0),
        (1.0, 0.8413),
        (2.0, 1.9545),
        (-1.0, -0.1587),
        (-2.0, -0.0455),
    ];

    for (x, expected) in test_cases {
        let input = Tensor::<B, D>::from_slice(&backend, &[x], &[1]).unwrap();
        let layer = Gelu::new();
        let mut ctx = ForwardCtx::eval();
        let output = layer.forward(input, &mut ctx).unwrap();
        let result: f32 = output.item().unwrap();

        // Tanh approximation has some error vs exact GELU, allow ~1% tolerance
        assert!(
            (result - expected).abs() < 0.02,
            "GELU({}) = {}, expected ~{}",
            x,
            result,
            expected
        );
    }
}

#[test]
fn gelu_forward_preserves_shape() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 1.2).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 3, 4]).unwrap();

    let layer = Gelu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[2, 3, 4]);
}

#[test]
fn gelu_is_monotonically_increasing_for_positive_inputs() {
    let backend = Arc::new(CpuBackend::new());
    // Generate positive values from 0 to 3
    let input_data: Vec<f32> = (0..31).map(|i| i as f32 * 0.1).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &input_data, &[31]).unwrap();

    let layer = Gelu::new();
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let data = output.to_vec().unwrap();

    // Check monotonicity for positive inputs
    for i in 1..data.len() {
        assert!(
            data[i] >= data[i - 1],
            "GELU should be monotonically increasing for positive x: f({}) = {} < f({}) = {}",
            input_data[i - 1],
            data[i - 1],
            input_data[i],
            data[i]
        );
    }
}
