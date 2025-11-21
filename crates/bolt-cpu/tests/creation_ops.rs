use std::sync::Arc;

use bolt_core::{Error, Tensor};
use bolt_cpu::CpuBackend;

#[test]
fn ones_contiguous_f32() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::ones(&backend, &[2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert!(tensor.layout().is_contiguous());
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0f32; 6]);
}

#[test]
fn full_contiguous_i32() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::full(&backend, &[2, 2], 7).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.to_vec().unwrap(), vec![7i32; 4]);
}

#[test]
fn ones_like_preserves_strides_and_storage_is_fresh() {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::zeros(&backend, &[2, 4]).unwrap();
    let view = base.slice(1, 0, 4, 2).unwrap();
    assert_eq!(view.strides(), &[4, 2]);

    let derived = Tensor::<CpuBackend, f32>::ones_like(&view).unwrap();

    assert_eq!(derived.shape(), view.shape());
    assert_eq!(derived.strides(), view.strides());
    assert_eq!(derived.to_vec().unwrap(), vec![1.0f32; 4]);
    assert!(!Arc::ptr_eq(
        view.storage().block(),
        derived.storage().block()
    ));
}

#[test]
fn full_like_non_contiguous_f64() {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f64>::zeros(&backend, &[3, 3]).unwrap();
    let view = base.slice(1, 0, 3, 2).unwrap();
    let derived = Tensor::<CpuBackend, f64>::full_like(&view, 3.5).unwrap();

    assert_eq!(derived.shape(), view.shape());
    assert_eq!(derived.strides(), view.strides());
    assert_eq!(derived.to_vec().unwrap(), vec![3.5f64; 6]);
}

#[test]
fn arange_f32_positive_step() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::arange(&backend, 0.0, 5.0, 1.0).unwrap();

    assert_eq!(tensor.shape(), &[5]);
    assert_eq!(tensor.to_vec().unwrap(), vec![0.0f32, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arange_i32_negative_step() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::arange(&backend, 5, -1, -2).unwrap();

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.to_vec().unwrap(), vec![5, 3, 1]);
}

#[test]
fn arange_rejects_zero_step() {
    let backend = Arc::new(CpuBackend::new());
    let err = match Tensor::<CpuBackend, f32>::arange(&backend, 0.0, 1.0, 0.0) {
        Ok(_) => panic!("expected zero step to error"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidShape { .. }));
}

#[test]
fn arange_rejects_non_progressing_step() {
    let backend = Arc::new(CpuBackend::new());
    let err = match Tensor::<CpuBackend, i32>::arange(&backend, 0, 5, -1) {
        Ok(_) => panic!("expected non-progressing arange to error"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidShape { .. }));
}
