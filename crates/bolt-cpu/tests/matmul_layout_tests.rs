use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

#[test]
fn matmul_supports_transposed_views() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let lhs =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let lhs_t = lhs.transpose(0, 1)?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        &[2, 4],
    )?;

    let out = lhs_t.matmul(&rhs)?.to_vec()?;
    assert_eq!(
        out,
        vec![
            51.0, 56.0, 61.0, 66.0, 69.0, 76.0, 83.0, 90.0, 87.0, 96.0, 105.0, 114.0
        ]
    );
    Ok(())
}

#[test]
fn matmul_supports_sliced_views_with_offsets() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let data: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[4, 5])?;

    let lhs = base.slice(0, 1, 3, 1)?.slice(1, 1, 4, 1)?;
    let rhs =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    let out = lhs.matmul(&rhs)?.to_vec()?;
    assert_eq!(out, vec![67.0, 88.0, 112.0, 148.0]);
    Ok(())
}

#[test]
fn matmul_supports_f64_inputs() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let lhs = Tensor::<CpuBackend, f64>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let rhs = Tensor::<CpuBackend, f64>::from_slice(&backend, &[5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let out = lhs.matmul(&rhs)?.to_vec()?;

    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    Ok(())
}
