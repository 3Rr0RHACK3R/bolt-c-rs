use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Dropout;
use bolt_nn::{ForwardCtx, Module};
use bolt_rng::RngKey;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

/// Test that multiple dropout layers get independent keys (no collisions).
#[test]
fn multiple_dropout_layers_independent() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let x = Tensor::<B, D>::from_slice(&backend, &data, &[8, 8]).unwrap();

    let dropout1 = Dropout::new(0.5).unwrap();
    let dropout2 = Dropout::new(0.5).unwrap();

    let key = RngKey::from_seed(123);
    let mut ctx = ForwardCtx::train_with_key(key);

    // First dropout
    let y1 = dropout1.forward(x.clone(), &mut ctx).unwrap();

    // Second dropout (should get different key due to counter)
    let y2 = dropout2.forward(x.clone(), &mut ctx).unwrap();

    // They should produce different outputs (very unlikely to be identical)
    assert_ne!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

/// Test that same dropout layer called twice gets different keys.
#[test]
fn same_dropout_called_twice_independent() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let x = Tensor::<B, D>::from_slice(&backend, &data, &[8, 8]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let key = RngKey::from_seed(456);
    let mut ctx = ForwardCtx::train_with_key(key);

    let y1 = dropout.forward(x.clone(), &mut ctx).unwrap();
    let y2 = dropout.forward(x, &mut ctx).unwrap();

    // Should be different due to counter increment
    assert_ne!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

/// Test determinism: same seed + same call order = same results.
#[test]
fn multiple_dropout_deterministic() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout1 = Dropout::new(0.5).unwrap();
    let dropout2 = Dropout::new(0.5).unwrap();

    // First run
    let key1 = RngKey::from_seed(789);
    let mut ctx1 = ForwardCtx::train_with_key(key1);
    let y1a = dropout1.forward(x.clone(), &mut ctx1).unwrap();
    let y2a = dropout2.forward(x.clone(), &mut ctx1).unwrap();

    // Second run with same seed
    let key2 = RngKey::from_seed(789);
    let mut ctx2 = ForwardCtx::train_with_key(key2);
    let y1b = dropout1.forward(x.clone(), &mut ctx2).unwrap();
    let y2b = dropout2.forward(x.clone(), &mut ctx2).unwrap();

    // Should be identical (deterministic)
    assert_eq!(y1a.to_vec().unwrap(), y1b.to_vec().unwrap());
    assert_eq!(y2a.to_vec().unwrap(), y2b.to_vec().unwrap());
}

/// Test that counter ensures no collisions even with many layers.
#[test]
fn many_dropout_layers_no_collisions() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let x = Tensor::<B, D>::from_slice(&backend, &data, &[8, 8]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let key = RngKey::from_seed(999);
    let mut ctx = ForwardCtx::train_with_key(key);

    let mut outputs = Vec::new();
    for _ in 0..10 {
        let y = dropout.forward(x.clone(), &mut ctx).unwrap();
        outputs.push(y.to_vec().unwrap());
    }

    // All outputs should be different (very unlikely to have collisions)
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            assert_ne!(
                outputs[i], outputs[j],
                "Collision detected between outputs {} and {}",
                i, j
            );
        }
    }
}
