use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn one_hot_basic() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[0, 2, 1], &[3])?;
    let one_hot = Tensor::<B, D>::one_hot(&indices, 3)?;

    assert_eq!(one_hot.shape(), &[3, 3]);
    let values = one_hot.to_vec()?;
    #[rustfmt::skip]
    let expected: Vec<D> = vec![
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
    ];
    assert_eq!(values, expected);
    Ok(())
}

#[test]
fn one_hot_single_element() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[2], &[1])?;
    let one_hot = Tensor::<B, D>::one_hot(&indices, 5)?;

    assert_eq!(one_hot.shape(), &[1, 5]);
    let values = one_hot.to_vec()?;
    let expected: Vec<D> = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(values, expected);
    Ok(())
}

#[test]
fn one_hot_f64_output() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[1, 0], &[2])?;
    let one_hot = Tensor::<B, f64>::one_hot(&indices, 3)?;

    assert_eq!(one_hot.shape(), &[2, 3]);
    let values = one_hot.to_vec()?;
    let expected: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(values, expected);
    Ok(())
}

#[test]
fn one_hot_index_out_of_bounds() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[0, 3, 1], &[3])?;
    let result = Tensor::<B, D>::one_hot(&indices, 3);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn one_hot_negative_index() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[0, -1, 1], &[3])?;
    let result = Tensor::<B, D>::one_hot(&indices, 3);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn one_hot_all_zeros() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[0, 0, 0], &[3])?;
    let one_hot = Tensor::<B, D>::one_hot(&indices, 4)?;

    assert_eq!(one_hot.shape(), &[3, 4]);
    let values = one_hot.to_vec()?;
    #[rustfmt::skip]
    let expected: Vec<D> = vec![
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(values, expected);
    Ok(())
}

#[test]
fn one_hot_last_class() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let indices = Tensor::<B, i32>::from_slice(&backend, &[4, 4], &[2])?;
    let one_hot = Tensor::<B, D>::one_hot(&indices, 5)?;

    assert_eq!(one_hot.shape(), &[2, 5]);
    let values = one_hot.to_vec()?;
    let expected: Vec<D> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(values, expected);
    Ok(())
}
