mod support;

use bolt_core::error::Result;

use support::test_runtime;

#[test]
fn slice_permute_and_contiguous_preserve_values() -> Result<()> {
    let runtime = test_runtime();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let tensor = runtime.tensor_from_slice(&[3, 4], &data)?;
    let sliced = tensor.slice(1, 1, 4, 2)?;
    let transposed = sliced.transpose(0, 1)?;
    let dense = transposed.contiguous()?;
    let values = dense.to_vec::<f32>()?;
    assert_eq!(values, vec![1.0, 5.0, 9.0, 3.0, 7.0, 11.0]);
    Ok(())
}

#[test]
fn binary_ops_handle_non_contiguous_inputs() -> Result<()> {
    let runtime = test_runtime();
    let lhs = runtime.tensor_from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0])?;
    let rhs = runtime.tensor_from_slice(&[2, 2], &[5.0f32, 6.0, 7.0, 8.0])?;
    let lhs_view = lhs.transpose(0, 1)?; // make it non-contiguous
    let sum = lhs_view.add(&rhs)?;
    let expected: Vec<f32> = lhs_view
        .contiguous()? // materialize view for reference math
        .to_vec::<f32>()?
        .into_iter()
        .zip(rhs.to_vec::<f32>()?)
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(sum.to_vec::<f32>()?, expected);
    Ok(())
}

#[test]
fn reshape_requires_contiguous_layout() -> Result<()> {
    let runtime = test_runtime();
    let tensor = runtime.tensor_from_slice(&[2, 4], &[1i32, 2, 3, 4, 5, 6, 7, 8])?;
    let view = tensor.slice(1, 0, 4, 2)?;
    assert!(view.reshape(&[4]).is_err());
    let flattened = view.contiguous()?.reshape(&[4])?;
    assert_eq!(flattened.shape(), &[4]);
    assert_eq!(flattened.to_vec::<i32>()?, vec![1, 3, 5, 7]);
    Ok(())
}
