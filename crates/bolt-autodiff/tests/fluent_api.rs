use std::sync::Arc;

use bolt_autodiff::{Attach, Graph, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

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
fn test_fluent_api_param_input() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?;
    let b_data = Tensor::from_slice(&backend, &[4.0_f32, 5.0, 6.0], &[3])?;

    let a = a_data.attach(&graph).with_grad();
    let b = b_data.attach(&graph).no_grad();

    assert!(a.requires_grad()?);
    assert!(!b.requires_grad()?);

    let c = a.add(&b)?;
    let loss = c.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);

    assert!(grads.wrt(&b).is_none());

    Ok(())
}

#[test]
fn test_tensor_like_mixing_gradtensor_and_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let w_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let x_data = Tensor::from_slice(&backend, &[4.0_f32, 5.0], &[2])?;

    let w = graph.param(&w_data);

    // This should work: GradTensor.add(Tensor)
    let y = w.add(&x_data)?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_matmul_tensor_like() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let w_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let x_data = Tensor::from_slice(&backend, &[5.0_f32, 6.0, 7.0, 8.0], &[2, 2])?;

    let w = graph.param(&w_data);

    let y = w.matmul(&x_data)?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    // Gradient of sum(matmul(w, x)) w.r.t. w is grad_y @ x^T, grad_y = ones([2,2])
    // ones([2,2]) @ x^T = [[5+6, 7+8], [5+6, 7+8]] = [[11,15],[11,15]]
    let expected = vec![11.0, 15.0, 11.0, 15.0];
    assert_vec_approx_eq(&dw, &expected, 1e-6);

    Ok(())
}
