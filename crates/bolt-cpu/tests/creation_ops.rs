use std::sync::Arc;

use bolt_core::Error;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn ones_contiguous_f32() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, D>::ones(&backend, &[2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert!(tensor.layout().is_contiguous());
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0; 6]);
}

#[test]
fn full_contiguous_i32() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, i32>::full(&backend, &[2, 2], 7).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.to_vec().unwrap(), vec![7; 4]);
}

#[test]
fn ones_like_preserves_strides_and_storage_is_fresh() {
    let backend = Arc::new(B::new());
    let base = Tensor::<B, D>::zeros(&backend, &[2, 4]).unwrap();
    let view = base.slice(1, 0, 4, 2).unwrap();
    assert_eq!(view.strides(), &[4, 2]);

    let derived = Tensor::<B, D>::ones_like(&view).unwrap();

    assert_eq!(derived.shape(), view.shape());
    assert_eq!(derived.strides(), view.strides());
    assert_eq!(derived.to_vec().unwrap(), vec![1.0; 4]);
    assert!(!Arc::ptr_eq(
        view.storage().block(),
        derived.storage().block()
    ));
}

#[test]
fn full_like_non_contiguous_f64() {
    let backend = Arc::new(B::new());
    let base = Tensor::<B, f64>::zeros(&backend, &[3, 3]).unwrap();
    let view = base.slice(1, 0, 3, 2).unwrap();
    let derived = Tensor::<B, f64>::full_like(&view, 3.5).unwrap();

    assert_eq!(derived.shape(), view.shape());
    assert_eq!(derived.strides(), view.strides());
    assert_eq!(derived.to_vec().unwrap(), vec![3.5; 6]);
}

#[test]
fn arange_f32_positive_step() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, D>::arange(&backend, 0.0, 5.0, 1.0).unwrap();

    assert_eq!(tensor.shape(), &[5]);
    assert_eq!(tensor.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arange_i32_negative_step() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, i32>::arange(&backend, 5, -1, -2).unwrap();

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.to_vec().unwrap(), vec![5, 3, 1]);
}

#[test]
fn arange_rejects_zero_step() {
    let backend = Arc::new(B::new());
    let err = match Tensor::<B, D>::arange(&backend, 0.0, 1.0, 0.0) {
        Ok(_) => panic!("expected zero step to error"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidShape { .. }));
}

#[test]
fn arange_rejects_non_progressing_step() {
    let backend = Arc::new(B::new());
    let err = match Tensor::<B, i32>::arange(&backend, 0, 5, -1) {
        Ok(_) => panic!("expected non-progressing arange to error"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidShape { .. }));
}

#[test]
fn from_vec_and_into_vec_roundtrip() {
    let backend = Arc::new(B::new());
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<B, D>::from_vec(&backend, data.clone(), &[2, 2]).unwrap();
    let roundtrip = tensor.into_vec().unwrap();
    assert_eq!(roundtrip, data);
}

#[test]
fn from_vec_size_mismatch_errors() {
    let backend = Arc::new(B::new());
    let result = Tensor::<B, D>::from_vec(&backend, vec![1.0, 2.0], &[2, 2]);
    assert!(matches!(result, Err(Error::SizeMismatch { .. })));
}

#[test]
fn zeros_like_resets_offset_and_values() {
    let backend = Arc::new(B::new());
    let base = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
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
    let backend = Arc::new(B::new());
    let eye = Tensor::<B, D>::eye(&backend, 2, 3).unwrap();
    assert_eq!(eye.shape(), &[2, 3]);
    assert_eq!(eye.to_vec().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let identity = Tensor::<B, D>::identity(&backend, 3).unwrap();
    assert_eq!(identity.shape(), &[3, 3]);
    assert_eq!(
        identity.to_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn linspace_includes_end_point() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, D>::linspace(&backend, 0.0, 3.0, 4).unwrap();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn linspace_rejects_small_step_counts() {
    let backend = Arc::new(B::new());
    let result = Tensor::<B, D>::linspace(&backend, 0.0, 1.0, 1);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}

#[test]
fn logspace_supports_custom_base() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, D>::logspace(&backend, 0.0, 3.0, 4, 2.0).unwrap();
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn logspace_rejects_invalid_bases() {
    let backend = Arc::new(B::new());
    let result = Tensor::<B, D>::logspace(&backend, 0.0, 3.0, 4, 1.0);
    assert!(matches!(result, Err(Error::InvalidShape { .. })));
}

#[test]
fn uniform_random_f32() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, Some(42)).unwrap();
    let t2 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, Some(42)).unwrap();

    assert_eq!(t1.to_vec().unwrap(), t2.to_vec().unwrap());

    let data = t1.to_vec().unwrap();
    for v in data {
        assert!((0.0..1.0).contains(&v));
    }
}

#[test]
fn normal_random_f32() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, Some(123)).unwrap();
    let t2 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, Some(123)).unwrap();

    assert_eq!(t1.to_vec().unwrap(), t2.to_vec().unwrap());

    let data = t1.to_vec().unwrap();
    let mean: D = data.iter().sum::<D>() / data.len() as D;
    let var: D = data.iter().map(|x| (x - mean).powi(2)).sum::<D>() / data.len() as D;
    let std = var.sqrt();

    assert!((mean - 0.0).abs() < 0.5);
    assert!((std - 1.0).abs() < 0.5);
}

#[test]
fn uniform_different_seeds_yield_different_vectors() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, Some(42)).unwrap();
    let t2 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, Some(43)).unwrap();

    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn normal_different_seeds_yield_different_vectors() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, Some(123)).unwrap();
    let t2 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, Some(456)).unwrap();

    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn uniform_no_seed_yields_different_vectors() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, None).unwrap();
    let t2 = Tensor::<B, D>::uniform(&backend, &[100], 0.0, 1.0, None).unwrap();

    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn normal_no_seed_yields_different_vectors() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, None).unwrap();
    let t2 = Tensor::<B, D>::normal(&backend, &[100], 0.0, 1.0, None).unwrap();

    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn bernoulli_mask_deterministic_with_same_seed() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 0.5, Some(42)).unwrap();
    let t2 = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 0.5, Some(42)).unwrap();

    assert_eq!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn bernoulli_mask_different_seeds_produce_different_results() {
    let backend = Arc::new(B::new());
    let t1 = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 0.5, Some(42)).unwrap();
    let t2 = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 0.5, Some(43)).unwrap();

    assert_ne!(t1.to_vec().unwrap(), t2.to_vec().unwrap());
}

#[test]
fn bernoulli_mask_values_are_binary() {
    let backend = Arc::new(B::new());
    let tensor = Tensor::<B, D>::bernoulli_mask(&backend, &[1000], 0.5, Some(42)).unwrap();
    let data = tensor.to_vec().unwrap();

    for v in data {
        assert!(v == 0.0 || v == 1.0, "bernoulli_mask should only produce 0.0 or 1.0, got {v}");
    }
}

#[test]
fn bernoulli_mask_statistical_properties() {
    let backend = Arc::new(B::new());
    let p_keep = 0.7;
    let tensor = Tensor::<B, D>::bernoulli_mask(&backend, &[10000], p_keep, Some(123)).unwrap();
    let data = tensor.to_vec().unwrap();

    let ones_count = data.iter().filter(|&&v| v == 1.0).count();
    let actual_proportion = ones_count as f32 / data.len() as f32;

    assert!(
        (actual_proportion - p_keep).abs() < 0.05,
        "Expected proportion ~{p_keep}, got {actual_proportion}"
    );
}

#[test]
fn bernoulli_mask_edge_cases() {
    let backend = Arc::new(B::new());
    
    let zeros = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 0.0, Some(42)).unwrap();
    assert_eq!(zeros.to_vec().unwrap(), vec![0.0; 100]);

    let ones = Tensor::<B, D>::bernoulli_mask(&backend, &[100], 1.0, Some(42)).unwrap();
    assert_eq!(ones.to_vec().unwrap(), vec![1.0; 100]);
}

#[test]
fn bernoulli_mask_rejects_invalid_p_keep() {
    let backend = Arc::new(B::new());
    
    let result_negative = Tensor::<B, D>::bernoulli_mask(&backend, &[10], -0.1, Some(42));
    assert!(matches!(result_negative, Err(Error::OpError { .. })));

    let result_too_large = Tensor::<B, D>::bernoulli_mask(&backend, &[10], 1.1, Some(42));
    assert!(matches!(result_too_large, Err(Error::OpError { .. })));
}
