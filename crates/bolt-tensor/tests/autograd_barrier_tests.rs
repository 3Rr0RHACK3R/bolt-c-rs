use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

#[test]
fn argmax_errors_when_requires_grad_enabled() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    assert!(x.argmax(None, false).is_err());
    Ok(())
}

#[test]
fn argmax_allows_detached_tensors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    let idx = x.detach().argmax(None, false)?;
    assert_eq!(idx.to_vec()?, vec![2]);
    Ok(())
}

#[test]
fn float_to_int_cast_errors_when_requires_grad_enabled() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    assert!(x.cast::<i32>().is_err());
    Ok(())
}

#[test]
fn float_to_int_cast_allows_detached_tensors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    let y = x.detach().cast::<i32>()?;
    assert_eq!(y.to_vec()?, vec![1, 2, 3]);
    Ok(())
}

#[test]
fn float_to_float_cast_preserves_values_without_grad_tracking() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.25_f32, -2.5, 3.0], &[3])?;
    let y = x.cast::<f64>()?;
    let yv = y.to_vec()?;
    assert_eq!(yv.len(), 3);
    assert!((yv[0] - 1.25).abs() < 1e-12);
    assert!((yv[1] + 2.5).abs() < 1e-12);
    assert!((yv[2] - 3.0).abs() < 1e-12);
    Ok(())
}
