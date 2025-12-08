use std::sync::Arc;

use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_losses::{accuracy_top1, cross_entropy, cross_entropy_from_logits, mse, Reduction};

type B = CpuBackend;
type D = f32;

#[test]
fn mse_reductions() {
    let backend = Arc::new(CpuBackend::new());
    let pred = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 1.0, 2.0], &[3]).unwrap();

    let none = mse(&pred, &target, Reduction::None).unwrap().to_vec().unwrap();
    assert_eq!(none, vec![0.0, 1.0, 1.0]);

    let sum = mse(&pred, &target, Reduction::Sum).unwrap().item().unwrap();
    assert_eq!(sum, 2.0);

    let mean = mse(&pred, &target, Reduction::Mean).unwrap().item().unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn cross_entropy_logits() {
    let backend = Arc::new(CpuBackend::new());
    // Two samples, three classes
    let logits = Tensor::<B, D>::from_slice(&backend, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0], &[2, 3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

    let loss = cross_entropy_from_logits(&logits, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();

    // For logits [2, 0, 0] (3 classes), cross-entropy = -log(exp(2)/(exp(2)+2)) = ln(exp(2)+2) - 2
    let expected = (2.0f32.exp() + 2.0).ln() - 2.0; // ≈ 0.24027
    assert!((loss - expected).abs() < 1e-4);
}

#[test]
fn cross_entropy_probs() {
    let backend = Arc::new(CpuBackend::new());
    let probs = Tensor::<B, D>::from_slice(&backend, &[0.7, 0.2, 0.1, 0.1, 0.2, 0.7], &[2, 3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]).unwrap();

    let loss = cross_entropy(&probs, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();

    let expected = -((0.7f32.ln() + 0.7f32.ln()) / 2.0);
    assert!((loss - expected).abs() < 1e-5);
}

#[test]
fn accuracy_top1_basic() {
    let backend = Arc::new(CpuBackend::new());
    let logits = Tensor::<B, D>::from_slice(&backend, &[2.0, 1.0, 0.5, 0.1, 0.2, 3.0], &[2, 3]).unwrap();
    let targets = Tensor::<B, i32>::from_slice(&backend, &[0, 2], &[2]).unwrap();

    let acc = accuracy_top1(&logits, &targets).unwrap();
    assert!((acc - 1.0).abs() < 1e-6);
}

// Note: Empty batch (shape [0, num_classes]) is not currently supported by tensor creation,
// but the accuracy_top1 function includes a guard to prevent division by zero if empty batches
// are ever supported in the future.

