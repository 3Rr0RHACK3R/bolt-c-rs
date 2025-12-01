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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[3.0, 1.0, 2.0, 5.0, 4.0, 0.5],
        &[3, 2],
    )?;

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
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )?;
    let transposed = tensor.transpose(0, 1)?;

    let result = transposed.sum(None, false)?;
    let result_vec = result.to_vec()?;
    assert!((result_vec[0] - 10.0).abs() < 1e-6);
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
    assert!((result_vec[0] - 1.0483200e+06).abs() < 1.0);
    assert!((result_vec[1] - 1.9535040e+08).abs() < 1000.0);
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
