use std::sync::Arc;

use bolt_core::Result;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn sum_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.sum(None, false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 1);
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_all_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.sum(None, true)?;
    assert_eq!(result.shape(), &[1, 1]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 1);
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_axis_0_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let result = tensor.sum(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 2);
    assert!((result_vec[0] - 9.0).abs() < 1e-6);
    assert!((result_vec[1] - 12.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_axis_1_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let result = tensor.sum(Some(&[1]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 3);
    assert!((result_vec[0] - 3.0).abs() < 1e-6);
    assert!((result_vec[1] - 7.0).abs() < 1e-6);
    assert!((result_vec[2] - 11.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_axis_0_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let result = tensor.sum(Some(&[0]), true)?;
    assert_eq!(result.shape(), &[1, 2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 2);
    assert!((result_vec[0] - 9.0).abs() < 1e-6);
    assert!((result_vec[1] - 12.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_multi_axis_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.sum(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[3]);
    Ok(())
}

#[test]
fn sum_multi_axis_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.sum(Some(&[0, 2]), true)?;
    assert_eq!(result.shape(), &[1, 3, 1]);
    Ok(())
}

#[test]
fn sum_empty_axes_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.sum(Some(&[]), false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_f64() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f64>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.sum(None, false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 1);
    assert!((result_vec[0] - 10.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn sum_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[1, 2, 3, 4], &[2, 2])?;

    let result = tensor.sum(None, false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec, vec![10]);
    Ok(())
}

#[test]
fn sum_axis_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[1, 2, 3, 4, 5, 6], &[3, 2])?;

    let result = tensor.sum(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec, vec![9, 12]);
    Ok(())
}

#[test]
fn prod_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?;

    let result = tensor.prod(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 24.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn min_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0, 4.0, 3.0], &[4])?;

    let result = tensor.min(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn max_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0, 4.0, 3.0], &[4])?;

    let result = tensor.max(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 4.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn argmin_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0, 4.0, 3.0], &[4])?;

    let result = tensor.argmin(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 1);
    Ok(())
}

#[test]
fn argmax_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0, 4.0, 3.0], &[4])?;

    let result = tensor.argmax(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 2);
    Ok(())
}

#[test]
fn prod_axis_0_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0, 5.0], &[2, 2])?;

    let result = tensor.prod(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 8.0).abs() < 1e-6);
    assert!((result_vec[1] - 15.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn prod_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?;

    let result = tensor.prod(None, true)?;
    assert_eq!(result.shape(), &[1]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 24.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn min_axis_0_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5], &[3, 2])?;

    let result = tensor.min(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 2.0).abs() < 1e-6);
    assert!((result_vec[1] - 0.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn max_axis_1_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5], &[3, 2])?;

    let result = tensor.max(Some(&[1]), true)?;
    assert_eq!(result.shape(), &[3, 1]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 3.0).abs() < 1e-6);
    assert!((result_vec[1] - 5.0).abs() < 1e-6);
    assert!((result_vec[2] - 4.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn argmin_axis_0_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5], &[3, 2])?;

    let result = tensor.argmin(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 1);
    assert_eq!(result_vec[1], 2);
    Ok(())
}

#[test]
fn argmax_axis_1_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5], &[3, 2])?;

    let result = tensor.argmax(Some(&[1]), true)?;
    assert_eq!(result.shape(), &[3, 1]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 0);
    assert_eq!(result_vec[1], 1);
    assert_eq!(result_vec[2], 0);
    Ok(())
}

#[test]
fn sum_non_contiguous_transposed() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let transposed = tensor.transpose(0, 1)?;

    let result = transposed.sum(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_axis_specific_on_transposed() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let transposed = tensor.transpose(0, 1)?;

    let result_axis0 = transposed.sum(Some(&[0]), false)?;
    assert_eq!(result_axis0.shape(), &[2]);
    let result_vec_axis0 = result_axis0.to_vec()?;
    assert!(
        (result_vec_axis0[0] - 6.0).abs() < 1e-6,
        "Expected 6.0, got {}",
        result_vec_axis0[0]
    );
    assert!(
        (result_vec_axis0[1] - 15.0).abs() < 1e-6,
        "Expected 15.0, got {}",
        result_vec_axis0[1]
    );

    let result_axis1 = transposed.sum(Some(&[1]), false)?;
    assert_eq!(result_axis1.shape(), &[3]);
    let result_vec_axis1 = result_axis1.to_vec()?;
    assert!((result_vec_axis1[0] - 5.0).abs() < 1e-6);
    assert!((result_vec_axis1[1] - 7.0).abs() < 1e-6);
    assert!((result_vec_axis1[2] - 9.0).abs() < 1e-6);
    Ok(())
}

fn non_contiguous_strided_fixture(backend: &Arc<CpuBackend>) -> Result<Tensor<CpuBackend, f32>> {
    let base =
        Tensor::<CpuBackend, f32>::from_slice(backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let transposed = base.transpose(0, 1)?;
    transposed.slice(0, 0, 3, 2)
}

#[test]
fn sum_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.sum(Some(&[0]), false)?;
    let axis0_vec = axis0.to_vec()?;
    assert!((axis0_vec[0] - 4.0).abs() < 1e-6);
    assert!((axis0_vec[1] - 10.0).abs() < 1e-6);

    let axis1 = view.sum(Some(&[1]), false)?;
    let axis1_vec = axis1.to_vec()?;
    assert!((axis1_vec[0] - 5.0).abs() < 1e-6);
    assert!((axis1_vec[1] - 9.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn prod_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.prod(Some(&[0]), false)?;
    let axis0_vec = axis0.to_vec()?;
    assert!((axis0_vec[0] - 3.0).abs() < 1e-6);
    assert!((axis0_vec[1] - 24.0).abs() < 1e-6);

    let axis1 = view.prod(Some(&[1]), false)?;
    let axis1_vec = axis1.to_vec()?;
    assert!((axis1_vec[0] - 4.0).abs() < 1e-6);
    assert!((axis1_vec[1] - 18.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn max_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.max(Some(&[0]), false)?;
    let axis0_vec = axis0.to_vec()?;
    assert!((axis0_vec[0] - 3.0).abs() < 1e-6);
    assert!((axis0_vec[1] - 6.0).abs() < 1e-6);

    let axis1 = view.max(Some(&[1]), false)?;
    let axis1_vec = axis1.to_vec()?;
    assert!((axis1_vec[0] - 4.0).abs() < 1e-6);
    assert!((axis1_vec[1] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn min_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.min(Some(&[0]), false)?;
    let axis0_vec = axis0.to_vec()?;
    assert!((axis0_vec[0] - 1.0).abs() < 1e-6);
    assert!((axis0_vec[1] - 4.0).abs() < 1e-6);

    let axis1 = view.min(Some(&[1]), false)?;
    let axis1_vec = axis1.to_vec()?;
    assert!((axis1_vec[0] - 1.0).abs() < 1e-6);
    assert!((axis1_vec[1] - 3.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.mean(Some(&[0]), false)?;
    let axis0_vec = axis0.to_vec()?;
    assert!((axis0_vec[0] - 2.0).abs() < 1e-6);
    assert!((axis0_vec[1] - 5.0).abs() < 1e-6);

    let axis1 = view.mean(Some(&[1]), false)?;
    let axis1_vec = axis1.to_vec()?;
    assert!((axis1_vec[0] - 2.5).abs() < 1e-6);
    assert!((axis1_vec[1] - 4.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn argmax_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.argmax(Some(&[0]), false)?;
    let axis0_vec: Vec<i32> = axis0.to_vec()?;
    assert_eq!(axis0_vec, vec![1, 1]);

    let axis1 = view.argmax(Some(&[1]), false)?;
    let axis1_vec: Vec<i32> = axis1.to_vec()?;
    assert_eq!(axis1_vec, vec![1, 1]);
    Ok(())
}

#[test]
fn argmin_non_contiguous_axis_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let view = non_contiguous_strided_fixture(&backend)?;

    let axis0 = view.argmin(Some(&[0]), false)?;
    let axis0_vec: Vec<i32> = axis0.to_vec()?;
    assert_eq!(axis0_vec, vec![0, 0]);

    let axis1 = view.argmin(Some(&[1]), false)?;
    let axis1_vec: Vec<i32> = axis1.to_vec()?;
    assert_eq!(axis1_vec, vec![0, 0]);
    Ok(())
}

#[test]
fn prod_negative_values_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, -3.0, 4.0], &[3])?;

    let result = tensor.prod(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - (-24.0)).abs() < 1e-6);
    Ok(())
}

#[test]
fn min_with_negative_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, -5.0, 4.0, -1.0], &[4])?;

    let result = tensor.min(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - (-5.0)).abs() < 1e-6);
    Ok(())
}

#[test]
fn argmin_tie_first_occurrence() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0, 4.0, 1.0], &[4])?;

    let result = tensor.argmin(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 1);
    Ok(())
}

#[test]
fn sum_3d_multi_axis() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.sum(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 68.0).abs() < 1e-6);
    assert!((result_vec[1] - 100.0).abs() < 1e-6);
    assert!((result_vec[2] - 132.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn prod_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[2, 3, 4], &[3])?;

    let result = tensor.prod(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 24);
    Ok(())
}

#[test]
fn min_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[5, -3, 8, 1], &[4])?;

    let result = tensor.min(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], -3);
    Ok(())
}

#[test]
fn max_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[5, -3, 8, 1], &[4])?;

    let result = tensor.max(None, false)?;
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 8);
    Ok(())
}

#[test]
fn prod_multi_axis_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.prod(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 1.048_32e6).abs() < 1.0);
    assert!((result_vec[1] - 1.953_504e8).abs() < 1000.0);
    assert!((result_vec[2] - 3.0296852e+09).abs() < 100000.0);
    Ok(())
}

#[test]
fn prod_multi_axis_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.prod(Some(&[0, 2]), true)?;
    assert_eq!(result.shape(), &[1, 3, 1]);
    Ok(())
}

#[test]
fn min_multi_axis_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.min(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 1.0).abs() < 1e-6);
    assert!((result_vec[1] - 5.0).abs() < 1e-6);
    assert!((result_vec[2] - 9.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn max_multi_axis_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.max(Some(&[0, 2]), true)?;
    assert_eq!(result.shape(), &[1, 3, 1]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 16.0).abs() < 1e-6);
    assert!((result_vec[1] - 20.0).abs() < 1e-6);
    assert!((result_vec[2] - 24.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn argmin_multi_axis_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.argmin(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;

    assert_eq!(result_vec[0], 0);
    assert_eq!(result_vec[1], 0);
    assert_eq!(result_vec[2], 0);
    Ok(())
}

#[test]
fn argmax_multi_axis_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?;

    let result = tensor.argmax(Some(&[0, 2]), true)?;
    assert_eq!(result.shape(), &[1, 3, 1]);
    let result_vec = result.to_vec()?;

    assert_eq!(result_vec[0], 1);
    assert_eq!(result_vec[1], 1);
    assert_eq!(result_vec[2], 1);
    Ok(())
}

// Mean reduction tests
#[test]
fn mean_all_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.mean(None, false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 1);
    assert!((result_vec[0] - 2.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_all_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.mean(None, true)?;
    assert_eq!(result.shape(), &[1, 1]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 1);
    assert!((result_vec[0] - 2.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_axis_0_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let result = tensor.mean(Some(&[0]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 2);
    assert!((result_vec[0] - 3.0).abs() < 1e-6); // (1+3+5)/3 = 3
    assert!((result_vec[1] - 4.0).abs() < 1e-6); // (2+4+6)/3 = 4
    Ok(())
}

#[test]
fn mean_axis_1_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let result = tensor.mean(Some(&[1]), true)?;
    assert_eq!(result.shape(), &[3, 1]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 3);
    assert!((result_vec[0] - 1.5).abs() < 1e-6); // (1+2)/2 = 1.5
    assert!((result_vec[1] - 3.5).abs() < 1e-6); // (3+4)/2 = 3.5
    assert!((result_vec[2] - 5.5).abs() < 1e-6); // (5+6)/2 = 5.5
    Ok(())
}

#[test]
fn mean_multi_axis_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    )?;

    let result = tensor.mean(Some(&[0, 2]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 2);
    // Mean over axes 0 and 2 for a [2,2,2] tensor
    // For position [0] in result: mean of [1,2,5,6] = 3.5
    // For position [1] in result: mean of [3,4,7,8] = 5.5
    assert!((result_vec[0] - 3.5).abs() < 1e-6);
    assert!((result_vec[1] - 5.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_multi_axis_keepdims_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    )?;

    let result = tensor.mean(Some(&[0, 2]), true)?;
    assert_eq!(result.shape(), &[1, 2, 1]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec.len(), 2);
    assert!((result_vec[0] - 3.5).abs() < 1e-6);
    assert!((result_vec[1] - 5.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_f64() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f64>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.mean(None, false)?;
    assert_eq!(result.shape(), &[]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 2.5).abs() < 1e-10);
    Ok(())
}

#[test]
fn sum_negative_axis() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let result = tensor.sum(Some(&[-1]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 6.0).abs() < 1e-6);
    assert!((result_vec[1] - 15.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn sum_negative_multi_axis() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    )?;

    let result = tensor.sum(Some(&[-2, -1]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
    assert!((result_vec[1] - 26.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn mean_negative_axis_keepdims() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let result = tensor.mean(Some(&[-1]), true)?;
    assert_eq!(result.shape(), &[2, 1]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 1.5).abs() < 1e-6);
    assert!((result_vec[1] - 3.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn max_negative_axis() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 4.0, 2.0, 3.0, 6.0, 5.0], &[2, 3])?;

    let result = tensor.max(Some(&[-2]), false)?;
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 3.0).abs() < 1e-6);
    assert!((result_vec[1] - 6.0).abs() < 1e-6);
    assert!((result_vec[2] - 5.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn min_negative_axis() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[5, 2, 9, 1, 7, 3], &[2, 3])?;

    let result = tensor.min(Some(&[-1]), false)?;
    assert_eq!(result.shape(), &[2]);
    let result_vec = result.to_vec()?;
    assert_eq!(result_vec[0], 2);
    assert_eq!(result_vec[1], 1);
    Ok(())
}

#[test]
fn transpose_negative_both_axes() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let result = tensor.transpose(-2, -1)?;
    assert_eq!(result.shape(), &[3, 2]);
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 1.0).abs() < 1e-6);
    assert!((result_vec[1] - 4.0).abs() < 1e-6);
    assert!((result_vec[2] - 2.0).abs() < 1e-6);
    assert!((result_vec[3] - 5.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn permute_all_negative() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
    )?;

    let result = tensor.permute(&[-1, -2, -3])?;
    assert_eq!(result.shape(), &[2, 2, 2]);
    Ok(())
}

#[test]
fn negative_axis_out_of_bounds() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    assert!(tensor.sum(Some(&[-3]), false).is_err());
    assert!(tensor.transpose(-3, 0).is_err());
    Ok(())
}
