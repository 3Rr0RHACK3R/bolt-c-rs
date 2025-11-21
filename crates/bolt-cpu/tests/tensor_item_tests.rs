use std::sync::Arc;

use bolt_core::{error::Error, error::Result, tensor::Tensor};
use bolt_cpu::CpuBackend;

#[test]
fn item_returns_scalar_value() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[42.0f32], &[1])?;

    let value = tensor.item()?;

    assert_eq!(value, 42.0);
    Ok(())
}

#[test]
fn item_rejects_multi_element_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[1i32, 2], &[2])?;

    let err = tensor.item();

    assert!(matches!(err, Err(Error::InvalidShape { .. })));
    Ok(())
}

#[test]
fn item_preserves_dtype_for_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[7i32], &[1])?;

    let value = tensor.item()?;

    assert_eq!(value, 7i32);
    Ok(())
}

#[test]
fn item_handles_strided_scalar_views() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[4])?;
    let every_other = tensor.slice(0, 1, 4, 2)?; // [2.0, 4.0]
    let last = every_other.slice(0, 1, 2, 1)?; // [4.0]

    let value = last.item()?;

    assert_eq!(value, 4.0);
    Ok(())
}
