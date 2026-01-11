use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_losses::{
    Reduction, accuracy_top1, accuracy_topk, binary_cross_entropy, binary_cross_entropy_with_logits,
    cross_entropy, cross_entropy_from_logits, mae, mse,
};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn mse_reductions() {
    let backend = Arc::new(CpuBackend::new());
    let pred = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 1.0, 2.0], &[3]).unwrap();

    let none = mse(&pred, &target, Reduction::None)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(none, vec![0.0, 1.0, 1.0]);

    let sum = mse(&pred, &target, Reduction::Sum).unwrap().item().unwrap();
    assert_eq!(sum, 2.0);

    let mean = mse(&pred, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn mae_reductions() {
    let backend = Arc::new(CpuBackend::new());
    let pred = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 1.0, 2.0], &[3]).unwrap();

    let none = mae(&pred, &target, Reduction::None)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(none, vec![0.0, 1.0, 1.0]);

    let sum = mae(&pred, &target, Reduction::Sum).unwrap().item().unwrap();
    assert_eq!(sum, 2.0);

    let mean = mae(&pred, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();
    assert!((mean - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn cross_entropy_logits() {
    let backend = Arc::new(CpuBackend::new());
    // Two samples, three classes
    let logits =
        Tensor::<B, D>::from_slice(&backend, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0], &[2, 3]).unwrap();
    let target =
        Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

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
    let probs =
        Tensor::<B, D>::from_slice(&backend, &[0.7, 0.2, 0.1, 0.1, 0.2, 0.7], &[2, 3]).unwrap();
    let target =
        Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]).unwrap();

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
    let logits =
        Tensor::<B, D>::from_slice(&backend, &[2.0, 1.0, 0.5, 0.1, 0.2, 3.0], &[2, 3]).unwrap();
    let targets = Tensor::<B, i32>::from_slice(&backend, &[0, 2], &[2]).unwrap();

    let acc = accuracy_top1(&logits, &targets).unwrap();
    assert!((acc - 1.0).abs() < 1e-6);
}

#[test]
fn accuracy_topk_basic() {
    let backend = Arc::new(CpuBackend::new());
    // Sample 1: logits [2.0, 1.0, 0.5] -> top-1 is class 0, top-2 are classes [0, 1]
    // Sample 2: logits [0.1, 0.2, 3.0] -> top-1 is class 2, top-2 are classes [2, 1]
    let logits =
        Tensor::<B, D>::from_slice(&backend, &[2.0, 1.0, 0.5, 0.1, 0.2, 3.0], &[2, 3]).unwrap();
    let targets = Tensor::<B, i32>::from_slice(&backend, &[0, 2], &[2]).unwrap();

    // Top-1: both correct -> 100%
    let acc1 = accuracy_topk(&logits, &targets, 1).unwrap();
    assert!((acc1 - 1.0).abs() < 1e-6);

    // Top-2: both correct (targets are in top-2) -> 100%
    let acc2 = accuracy_topk(&logits, &targets, 2).unwrap();
    assert!((acc2 - 1.0).abs() < 1e-6);
}

#[test]
fn accuracy_topk_partial_match() {
    let backend = Arc::new(CpuBackend::new());
    // Sample 1: logits [2.0, 1.0, 0.5] -> top-1 is class 0, top-2 are classes [0, 1]
    // Sample 2: logits [0.1, 0.2, 3.0] -> top-1 is class 2, top-2 are classes [2, 1]
    let logits =
        Tensor::<B, D>::from_slice(&backend, &[2.0, 1.0, 0.5, 0.1, 0.2, 3.0], &[2, 3]).unwrap();
    // Target 1 is class 1 (not top-1, but in top-2), target 2 is class 2 (top-1)
    let targets = Tensor::<B, i32>::from_slice(&backend, &[1, 2], &[2]).unwrap();

    // Top-1: only second sample correct -> 50%
    let acc1 = accuracy_topk(&logits, &targets, 1).unwrap();
    assert!((acc1 - 0.5).abs() < 1e-6);

    // Top-2: both correct -> 100%
    let acc2 = accuracy_topk(&logits, &targets, 2).unwrap();
    assert!((acc2 - 1.0).abs() < 1e-6);
}

#[test]
fn accuracy_topk_top5() {
    let backend = Arc::new(CpuBackend::new());
    // 3 samples, 10 classes
    // Sample 1: highest at index 5
    // Sample 2: highest at index 3
    // Sample 3: highest at index 7
    let mut logits_data = vec![0.0f32; 3 * 10];
    logits_data[5] = 10.0; // Sample 1, class 5
    logits_data[13] = 10.0; // Sample 2, class 3
    logits_data[27] = 10.0; // Sample 3, class 7
    
    let logits = Tensor::<B, D>::from_slice(&backend, &logits_data, &[3, 10]).unwrap();
    let targets = Tensor::<B, i32>::from_slice(&backend, &[5, 3, 7], &[3]).unwrap();

    // Top-5: all targets should be in top-5 -> 100%
    let acc5 = accuracy_topk(&logits, &targets, 5).unwrap();
    assert!((acc5 - 1.0).abs() < 1e-6);
}

#[test]
fn binary_cross_entropy_basic() {
    let backend = Arc::new(CpuBackend::new());
    let pred = Tensor::<B, D>::from_slice(&backend, &[0.7, 0.3, 0.9], &[3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 1.0], &[3]).unwrap();

    let loss = binary_cross_entropy(&pred, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();

    let expected = -((0.7f32.ln() + (1.0 - 0.3f32).ln() + 0.9f32.ln()) / 3.0);
    assert!((loss - expected).abs() < 1e-4);
}

#[test]
fn binary_cross_entropy_reductions() {
    let backend = Arc::new(CpuBackend::new());
    let pred = Tensor::<B, D>::from_slice(&backend, &[0.8, 0.2], &[2]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0], &[2]).unwrap();

    let none = binary_cross_entropy(&pred, &target, Reduction::None)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(none.len(), 2);

    let sum = binary_cross_entropy(&pred, &target, Reduction::Sum)
        .unwrap()
        .item()
        .unwrap();
    assert!(sum > 0.0);

    let mean = binary_cross_entropy(&pred, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();
    assert!((mean - sum / 2.0).abs() < 1e-5);
}

#[test]
fn binary_cross_entropy_with_logits_basic() {
    let backend = Arc::new(CpuBackend::new());
    let logits = Tensor::<B, D>::from_slice(&backend, &[2.0, -1.0, 1.5], &[3]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 1.0], &[3]).unwrap();

    let loss = binary_cross_entropy_with_logits(&logits, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn binary_cross_entropy_with_logits_reductions() {
    let backend = Arc::new(CpuBackend::new());
    let logits = Tensor::<B, D>::from_slice(&backend, &[1.0, -1.0], &[2]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0], &[2]).unwrap();

    let none = binary_cross_entropy_with_logits(&logits, &target, Reduction::None)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(none.len(), 2);
    assert!(none.iter().all(|&x| x.is_finite() && x >= 0.0));

    let sum = binary_cross_entropy_with_logits(&logits, &target, Reduction::Sum)
        .unwrap()
        .item()
        .unwrap();
    assert!(sum > 0.0);
    assert!(sum.is_finite());

    let mean = binary_cross_entropy_with_logits(&logits, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();
    assert!((mean - sum / 2.0).abs() < 1e-5);
}

#[test]
fn binary_cross_entropy_with_logits_numerical_stability() {
    let backend = Arc::new(CpuBackend::new());
    let large_logits = Tensor::<B, D>::from_slice(&backend, &[100.0, -100.0], &[2]).unwrap();
    let target = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0], &[2]).unwrap();

    let loss = binary_cross_entropy_with_logits(&large_logits, &target, Reduction::Mean)
        .unwrap()
        .item()
        .unwrap();

    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}
