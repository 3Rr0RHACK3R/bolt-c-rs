use std::sync::Arc;

use bolt_autodiff::{Graph, Result};
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
fn test_backward_on_intermediate_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let w_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let w = graph.param(&w_data);
    let y = w.mul(&w)?;

    // z is the final tensor, but we won't use it for backward
    let _z = y.mul(&w)?;

    // Call backward on the intermediate tensor y
    let grads = graph.backward(&y)?;

    // Gradient of y = w*w w.r.t w is 2*w (element-wise)
    // Since backward on a non-scalar is an implicit sum, the grad is sum(2*w)
    // No, the grad of y w.r.t w is 2w. The VJP is with a vector of 1s.
    // So d(w1^2)/dw1 = 2*w1, d(w2^2)/dw2 = 2*w2
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[4.0, 6.0], 1e-6);

    Ok(())
}

#[test]
fn test_backward_on_non_scalar_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let w_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let c_data = Tensor::from_slice(&backend, &[10.0_f32, 100.0], &[2])?;
    let w = graph.param(&w_data);
    let c = graph.input(&c_data);

    // y is a non-scalar tensor [20.0, 300.0]
    let y = w.mul(&c)?;

    // Call backward on the non-scalar tensor y.
    // This is equivalent to backward on y.sum()
    let grads = graph.backward(&y)?;

    // L = y1 + y2 = w1*c1 + w2*c2
    // dL/dw = [c1, c2]
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[10.0, 100.0], 1e-6);

    Ok(())
}
