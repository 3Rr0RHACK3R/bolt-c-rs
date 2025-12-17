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

#[test]
fn test_no_accumulate_for_distinct_untracked_inputs() -> Result<()> {
    // Regression test: ensure gradients for different untracked inputs (Handle::NONE)
    // are not accumulated together, which previously caused shape-mismatch errors.
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    // Shapes
    let batch = 4usize;
    let in_dim = 5usize;
    let hidden = 3usize;
    let classes = 2usize;

    // Two distinct untracked inputs with different shapes: x [B,in], t [B,classes]
    let x_data = vec![1.0_f32; batch * in_dim];
    let t_data = vec![0.5_f32; batch * classes];
    let x = Tensor::from_slice(&autodiff, &x_data, &[batch, in_dim])?; // NOT requires_grad
    let t = Tensor::from_slice(&autodiff, &t_data, &[batch, classes])?; // NOT requires_grad

    // Trainable weights
    // w1: [in_dim, hidden], w2: [hidden, classes]
    let w1 = Tensor::from_slice(
        &autodiff,
        &vec![0.1_f32; in_dim * hidden],
        &[in_dim, hidden],
    )?
    .requires_grad();
    let w2 = Tensor::from_slice(
        &autodiff,
        &vec![0.2_f32; hidden * classes],
        &[hidden, classes],
    )?
    .requires_grad();

    // Forward: y = (x @ w1) @ w2
    let h = x.matmul(&w1)?; // [B, hidden]
    let y = h.matmul(&w2)?; // [B, classes]

    // Loss ~ MSE(y, t): mean((y - t)^2)
    let diff = y.sub(&t)?;
    let sq = diff.mul(&diff)?;
    let loss = sq.mean(None, false)?; // scalar

    // Backward should succeed and produce grads for w1 and w2 without shape mismatch.
    let grads = loss.backward()?;
    assert!(grads.wrt(&w1).is_some());
    assert!(grads.wrt(&w2).is_some());
    assert!(grads.wrt(&x).is_none()); // x is untracked
    assert!(grads.wrt(&t).is_none()); // t is untracked

    Ok(())
}
