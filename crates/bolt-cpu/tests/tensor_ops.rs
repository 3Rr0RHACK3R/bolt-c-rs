use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

#[test]
fn add_sub_matmul() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2])?;

    let sum = a.add(&b)?.to_vec()?;
    assert_eq!(sum, vec![11.0, 22.0, 33.0, 44.0]);

    let diff = b.sub(&a)?.to_vec()?;
    assert_eq!(diff, vec![9.0, 18.0, 27.0, 36.0]);

    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0, 3.0, 2.0, 1.0], &[2, 2])?;
    let prod = lhs.matmul(&rhs)?.to_vec()?;
    assert_eq!(prod, vec![8.0, 5.0, 20.0, 13.0]);

    Ok(())
}

#[test]
fn broadcasting_add() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, -1.0], &[1, 2])?;

    let result = a.add(&b)?.to_vec()?;
    assert_eq!(result, vec![2.0, 1.0, 4.0, 3.0]);
    Ok(())
}

#[test]
fn non_contiguous_view_ops() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let permuted = base.permute(&[1, 0])?;
    let other =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2])?;

    let result = permuted.add(&other)?.to_vec()?;
    assert_eq!(result, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]);
    Ok(())
}

#[test]
fn mean_full_reduction() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 3.0, 5.0, 7.0], &[2, 2])?;
    let mean = tensor.mean(None, false)?.to_vec()?;
    assert_eq!(mean, vec![4.0]);
    Ok(())
}

#[test]
fn mean_handles_f64_inputs() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f64>::from_slice(&backend, &[1.0, 3.0, 5.0, 7.0], &[2, 2])?;
    let mean = tensor.mean(None, false)?.to_vec()?;
    assert_eq!(mean, vec![4.0]);
    Ok(())
}

#[test]
fn squeeze_and_unsqueeze_round_trip() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;

    let squeezed = tensor.squeeze()?;
    assert_eq!(squeezed.shape(), &[2, 2]);

    let axis_squeezed = tensor.squeeze_axis(0)?;
    assert_eq!(axis_squeezed.shape(), &[2, 2]);

    let unsqueezed = squeezed.unsqueeze(-1)?;
    assert_eq!(unsqueezed.shape(), &[2, 2, 1]);

    let restored = unsqueezed.squeeze()?;
    assert_eq!(restored.shape(), &[2, 2]);
    Ok(())
}

#[test]
fn unsqueeze_negative_axis_inserts_front() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?;
    let unsqueezed = tensor.unsqueeze(-2)?;
    assert_eq!(unsqueezed.shape(), &[1, 3]);
    let expanded = unsqueezed.expand(&[2, -1])?;
    assert_eq!(expanded.shape(), &[2, 3]);
    assert_eq!(expanded.to_vec()?, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn expand_supports_negative_one() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 1, 2])?;
    let expanded = tensor.expand(&[2, 3, -1])?;
    assert_eq!(expanded.shape(), &[2, 3, 2]);
    let values = expanded.to_vec()?;
    assert_eq!(
        values,
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );
    Ok(())
}
