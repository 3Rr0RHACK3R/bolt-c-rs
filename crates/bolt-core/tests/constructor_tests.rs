use std::sync::Arc;

use bolt_core::{Error, Tensor};
use bolt_cpu::CpuBackend;

fn backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

#[test]
fn from_vec_and_into_vec_roundtrip() {
    let backend = backend();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CpuBackend, f32>::from_vec(&backend, data.clone(), &[2, 2]).unwrap();
    let roundtrip = tensor.into_vec().unwrap();
    assert_eq!(roundtrip, data);
}

#[test]
fn from_vec_size_mismatch_errors() {
    let backend = backend();
    let result = Tensor::<CpuBackend, f32>::from_vec(&backend, vec![1.0, 2.0], &[2, 2]);
    assert!(matches!(result, Err(Error::SizeMismatch { .. })));
}

#[test]
fn zeros_like_resets_offset_and_values() {
    let backend = backend();
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
    let backend = backend();
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
    let backend = backend();
    let tensor = Tensor::<CpuBackend, f32>::linspace(&backend, 0.0, 3.0, 4).unwrap();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn linspace_rejects_small_step_counts() {
    let backend = backend();
    let result = Tensor::<CpuBackend, f32>::linspace(&backend, 0.0, 1.0, 1);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}

#[test]
fn logspace_supports_custom_base() {
    let backend = backend();
    let tensor = Tensor::<CpuBackend, f32>::logspace(&backend, 0.0, 3.0, 4, 2.0).unwrap();
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn logspace_rejects_invalid_bases() {
    let backend = backend();
    let result = Tensor::<CpuBackend, f32>::logspace(&backend, 0.0, 3.0, 4, 1.0);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}
