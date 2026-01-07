use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Dropout;
use bolt_nn::{ForwardCtx, Module};
use bolt_rng::RngKey;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn dropout_eval_is_identity() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = dropout.forward(x.clone(), &mut ctx).unwrap();
    assert_eq!(y.to_vec().unwrap(), x.to_vec().unwrap());
}

#[test]
fn dropout_same_seed_same_output() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();

    let key1 = RngKey::from_seed(123);
    let mut ctx1 = ForwardCtx::train_with_key(key1);
    let y1 = dropout.forward(x.clone(), &mut ctx1).unwrap();

    let key2 = RngKey::from_seed(123);
    let mut ctx2 = ForwardCtx::train_with_key(key2);
    let y2 = dropout.forward(x, &mut ctx2).unwrap();

    assert_eq!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

#[test]
fn dropout_advances_rng_between_calls() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let key = RngKey::from_seed(999);
    let mut ctx = ForwardCtx::train_with_key(key);

    let y1 = dropout.forward(x.clone(), &mut ctx).unwrap();
    let y2 = dropout.forward(x, &mut ctx).unwrap();

    assert_ne!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

#[test]
fn dropout_train_ctx_has_default_rngs() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let key = RngKey::from_seed(42);
    let mut ctx = ForwardCtx::train_with_key(key);
    dropout.forward(x, &mut ctx).unwrap();
}

#[test]
fn dropout_rate_statistics() {
    let backend = Arc::new(CpuBackend::new());
    let mut base_key = RngKey::from_seed(12345);

    let p = 0.5;
    let dropout = Dropout::new(p).unwrap();

    let num_samples = 1000;
    let tensor_size = 100;
    let mut total_zeroed = 0;
    let total_elements = num_samples * tensor_size;

    for i in 0..num_samples {
        let x_data: Vec<f32> = (0..tensor_size)
            .map(|j| (i * tensor_size + j) as f32)
            .collect();
        let x = Tensor::<B, D>::from_slice(&backend, &x_data, &[tensor_size]).unwrap();

        let key = {
            let (k, next) = base_key.split();
            base_key = next;
            k
        };
        let mut ctx = ForwardCtx::train_with_key(key);
        let y = dropout.forward(x, &mut ctx).unwrap();
        let y_vec = y.to_vec().unwrap();

        for val in y_vec {
            if val.abs() < 1e-6 {
                total_zeroed += 1;
            }
        }
    }

    let observed_rate = total_zeroed as f64 / total_elements as f64;
    let tolerance = 0.05;
    assert!(
        (observed_rate - p).abs() < tolerance,
        "Dropout rate mismatch: expected {p}, observed {observed_rate} (tolerance: {tolerance})"
    );
}

#[test]
fn dropout_mean_preservation() {
    let backend = Arc::new(CpuBackend::new());
    let p = 0.5;
    let dropout = Dropout::new(p).unwrap();

    let num_samples = 1000;
    let tensor_size = 100;
    let mut total_input_sum = 0.0;
    let mut total_output_sum = 0.0;
    let mut total_elements = 0;

    for i in 0..num_samples {
        let x_data: Vec<f32> = (0..tensor_size)
            .map(|j| (i * tensor_size + j) as f32)
            .collect();
        let x = Tensor::<B, D>::from_slice(&backend, &x_data, &[tensor_size]).unwrap();

        let input_mean = x.mean(None, false).unwrap();
        let input_mean_val: f64 = input_mean.item().unwrap().into();

        let key = RngKey::from_seed(i as u64);
        let mut ctx = ForwardCtx::train_with_key(key);
        let y = dropout.forward(x, &mut ctx).unwrap();

        let output_mean = y.mean(None, false).unwrap();
        let output_mean_val: f64 = output_mean.item().unwrap().into();

        total_input_sum += input_mean_val;
        total_output_sum += output_mean_val;
        total_elements += 1;
    }

    let avg_input_mean = total_input_sum / total_elements as f64;
    let avg_output_mean = total_output_sum / total_elements as f64;
    let relative_error = (avg_output_mean - avg_input_mean).abs() / avg_input_mean.abs().max(1e-6);

    let tolerance = 0.1;
    assert!(
        relative_error < tolerance,
        "Mean preservation failed: input_mean={avg_input_mean}, output_mean={avg_output_mean}, relative_error={relative_error} (tolerance: {tolerance})"
    );
}

#[test]
fn dropout_edge_cases() {
    let backend = Arc::new(CpuBackend::new());
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = Tensor::<B, D>::from_slice(&backend, &x_data, &[5]).unwrap();

    let dropout_p0 = Dropout::new(0.0).unwrap();
    let key0 = RngKey::from_seed(42);
    let mut ctx = ForwardCtx::train_with_key(key0);
    let y_p0 = dropout_p0.forward(x.clone(), &mut ctx).unwrap();
    assert_eq!(
        y_p0.to_vec().unwrap(),
        x.to_vec().unwrap(),
        "p=0.0 should be identity"
    );

    let dropout_p1 = Dropout::new(1.0).unwrap();
    let key1 = RngKey::from_seed(42);
    let mut ctx = ForwardCtx::train_with_key(key1);
    let y_p1 = dropout_p1.forward(x, &mut ctx).unwrap();
    let y_p1_vec = y_p1.to_vec().unwrap();
    for val in y_p1_vec {
        assert!(val.abs() < 1e-6, "p=1.0 should output all zeros, got {val}");
    }
}
