mod support;

use bolt_core::error::{Error, Result};

use support::test_runtime;

#[test]
fn item_returns_scalar_value() -> Result<()> {
    let runtime = test_runtime();
    let tensor = runtime.tensor_from_slice(&[1], &[42.0f32])?;
    let value: f32 = tensor.item()?;
    assert_eq!(value, 42.0);
    Ok(())
}

#[test]
fn item_rejects_multi_element_tensor() -> Result<()> {
    let runtime = test_runtime();
    let tensor = runtime.tensor_from_slice(&[2], &[1i32, 2])?;
    let err = tensor.item::<i32>();
    assert!(matches!(err, Err(Error::InvalidShape { .. })));
    Ok(())
}

#[test]
fn item_rejects_dtype_mismatch() -> Result<()> {
    let runtime = test_runtime();
    let tensor = runtime.tensor_from_slice(&[1], &[7i32])?;
    let err = tensor.item::<f32>();
    assert!(matches!(
        err,
        Err(Error::DTypeMismatch { lhs, rhs }) if lhs == tensor.dtype() && rhs != lhs
    ));
    Ok(())
}

#[test]
fn item_handles_strided_scalar_views() -> Result<()> {
    let runtime = test_runtime();
    let tensor = runtime.tensor_from_slice(&[4], &[1.0f32, 2.0, 3.0, 4.0])?;
    let view = tensor.slice(0, 1, 4, 2)?; // picks indices 1 and 3 -> size 2
    let last = view.slice(0, 1, 2, 1)?; // narrow to the element sourced from offset 3
    let value: f32 = last.item()?;
    assert_eq!(value, 4.0);
    Ok(())
}
