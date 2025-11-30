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

#[test]
fn from_vec_and_into_vec_roundtrip() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CpuBackend, f32>::from_vec(&backend, data.clone(), &[2, 2]).unwrap();
    let roundtrip = tensor.into_vec().unwrap();
    assert_eq!(roundtrip, data);
}

#[test]
fn from_vec_size_mismatch_errors() {
    let backend = Arc::new(CpuBackend::new());
    let result = Tensor::<CpuBackend, f32>::from_vec(&backend, vec![1.0, 2.0], &[2, 2]);
    assert!(matches!(result, Err(Error::SizeMismatch { .. })));
}

#[test]
fn zeros_like_resets_offset_and_values() {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let view = base.slice(0, 1, 2, 1).unwrap();
    assert!(view.layout().offset_bytes() > 0);

    let zeros = Tensor::zeros_like(&view).unwrap();
    assert_eq!(zeros.shape(), view.shape());
    assert!(zeros.layout().is_contiguous());
    assert_eq!(zeros.layout().offset_bytes(), 0);
    assert_eq!(zeros.to_vec().unwrap(), vec![0.0, 0.0]);
}

#[test]
fn eye_and_identity_match_expected_patterns() {
    let backend = Arc::new(CpuBackend::new());
    let eye = Tensor::<CpuBackend, f32>::eye(&backend, 2, 3).unwrap();
    assert_eq!(eye.shape(), &[2, 3]);
    assert_eq!(eye.to_vec().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let identity = Tensor::<CpuBackend, f32>::identity(&backend, 3).unwrap();
    assert_eq!(identity.shape(), &[3, 3]);
    assert_eq!(
        identity.to_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn linspace_includes_end_point() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::linspace(&backend, 0.0, 3.0, 4).unwrap();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn linspace_rejects_small_step_counts() {
    let backend = Arc::new(CpuBackend::new());
    let result = Tensor::<CpuBackend, f32>::linspace(&backend, 0.0, 1.0, 1);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}

#[test]
fn logspace_supports_custom_base() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::logspace(&backend, 0.0, 3.0, 4, 2.0).unwrap();
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn logspace_rejects_invalid_bases() {
    let backend = Arc::new(CpuBackend::new());
    let result = Tensor::<CpuBackend, f32>::logspace(&backend, 0.0, 3.0, 4, 1.0);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}
