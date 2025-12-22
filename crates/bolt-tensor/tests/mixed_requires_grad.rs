use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(*a, *e, eps), "idx {i}: got {a}, expected {e}");
    }
}

#[test]
fn mixing_requires_grad_and_non_requires_grad_only_returns_leaf_grads() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0_f32, 5.0, 6.0], &[3])?;

    let loss = a.add(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let da = grads.wrt(&a).unwrap().to_vec()?;
    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);
    assert!(grads.wrt(&b).is_none());

    Ok(())
}

#[test]
fn distinct_non_requires_grad_inputs_do_not_interfere_with_grad_shapes() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let batch = 4usize;
    let in_dim = 5usize;
    let hidden = 3usize;
    let classes = 2usize;

    let x_data = vec![1.0_f32; batch * in_dim];
    let t_data = vec![0.5_f32; batch * classes];
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &x_data, &[batch, in_dim])?;
    let t = Tensor::<CpuBackend, f32>::from_slice(&backend, &t_data, &[batch, classes])?;

    let w1 = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &vec![0.1_f32; in_dim * hidden],
        &[in_dim, hidden],
    )?
    .requires_grad();
    let w2 = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &vec![0.2_f32; hidden * classes],
        &[hidden, classes],
    )?
    .requires_grad();

    let h = x.matmul(&w1)?;
    let y = h.matmul(&w2)?;
    let diff = y.sub(&t)?;
    let sq = diff.mul(&diff)?;
    let loss = sq.mean(None, false)?;

    let grads = loss.backward()?;
    assert!(grads.wrt(&w1).is_some());
    assert!(grads.wrt(&w2).is_some());
    assert!(grads.wrt(&x).is_none());
    assert!(grads.wrt(&t).is_none());

    Ok(())
}
