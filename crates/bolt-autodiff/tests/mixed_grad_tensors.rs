use bolt_autodiff::{Autodiff, AutodiffTensorExt, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, eps),
            "element {} differs: got {}, expected {} (eps={})",
            i,
            a,
            e,
            eps
        );
    }
}

#[test]
fn test_mixing_tracked_and_untracked_tensors() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let a = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0], &[3])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[4.0_f32, 5.0, 6.0], &[3])?;

    assert!(a.is_tracked());
    assert!(!b.is_tracked());

    let c = a.add(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);

    assert!(grads.wrt(&b).is_none());

    Ok(())
}
