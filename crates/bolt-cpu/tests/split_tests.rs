use std::sync::Arc;

use bolt_core::{
    SplitSpec,
    error::{Error, Result},
    runtime::Runtime,
};
use bolt_cpu::CpuRuntimeBuilderExt;

#[test]
fn split_chunk_even_sections() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[4usize], &[1.0f32, 2.0, 3.0, 4.0])?;
    let outputs = tensor.split(SplitSpec::ChunkSize(2), 0)?;
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[0].to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(outputs[1].to_vec::<f32>()?, vec![3.0, 4.0]);
    Ok(())
}

#[test]
fn split_chunk_with_remainder() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[5usize], &[1i32, 2, 3, 4, 5])?;
    let outputs = tensor.split(SplitSpec::ChunkSize(2), 0)?;
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[1]);
    assert_eq!(outputs[2].to_vec::<i32>()?, vec![5]);
    Ok(())
}

#[test]
fn split_sections_exact_cover() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[4usize], &[1.0f32, 2.0, 3.0, 4.0])?;
    let outputs = tensor.split(SplitSpec::Sections(vec![1, 2, 1]), 0)?;
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[1].to_vec::<f32>()?, vec![2.0, 3.0]);
    Ok(())
}

#[test]
fn split_sections_sum_mismatch_errors() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[4usize], &[1.0f32, 2.0, 3.0, 4.0])?;
    let err = match tensor.split(SplitSpec::Sections(vec![1, 1]), 0) {
        Ok(_) => panic!("sections sum mismatch should error"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        Error::SizeMismatch {
            expected: 4,
            actual: 2
        }
    ));
    Ok(())
}

#[test]
fn split_negative_axis() -> Result<()> {
    let runtime = cpu_runtime()?;
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let tensor = runtime.tensor_from_slice(&[2usize, 4], &data)?;
    let outputs = tensor.split(SplitSpec::ChunkSize(2), -1)?;
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[0].to_vec::<f32>()?, vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(outputs[1].to_vec::<f32>()?, vec![3.0, 4.0, 7.0, 8.0]);
    Ok(())
}

#[test]
fn split_preserves_strides_for_non_contiguous_input() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[2usize, 3], &[1i32, 2, 3, 4, 5, 6])?;
    let view = tensor.transpose(0, 1)?;
    assert!(!view.is_contiguous());
    let outputs = view.split(SplitSpec::ChunkSize(1), 0)?;
    assert_eq!(outputs.len(), 3);
    for out in &outputs {
        assert_eq!(out.device_kind(), view.device_kind());
        assert_eq!(out.dtype(), view.dtype());
        assert_eq!(out.buffer_id(), view.buffer_id());
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.strides()[1], view.strides()[1]);
    }
    Ok(())
}

#[test]
fn split_rejects_invalid_specs() -> Result<()> {
    let runtime = cpu_runtime()?;
    let tensor = runtime.tensor_from_slice(&[4usize], &[1.0f32, 2.0, 3.0, 4.0])?;
    let err = match tensor.split(SplitSpec::ChunkSize(0), 0) {
        Ok(_) => panic!("chunk size zero is invalid"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidShape { .. }));

    let err = match tensor.split(SplitSpec::Sections(vec![1, 0, 3]), 0) {
        Ok(_) => panic!("zero-sized section is invalid"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidShape { .. }));
    Ok(())
}

fn cpu_runtime() -> Result<Arc<Runtime>> {
    Runtime::builder().with_cpu()?.build()
}
