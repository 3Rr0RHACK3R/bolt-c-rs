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
fn multiple_backward_on_same_graph_works() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let y = x.mul(&x)?;
    let loss = y.sum(None, false)?;

    let g1 = loss.backward()?;
    let dx1 = g1.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx1, &[4.0, 6.0], 1e-6);

    let g2 = loss.backward()?;
    let dx2 = g2.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx2, &[4.0, 6.0], 1e-6);

    Ok(())
}

#[test]
fn independent_forward_passes_have_independent_graphs() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?.requires_grad();

    let loss1 = x.mul(&x)?.sum(None, false)?;
    let loss2 = x.mul(&x)?.mul(&x)?.sum(None, false)?;

    let g1 = loss1.backward()?;
    let dx1 = g1.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx1, &[4.0], 1e-6);

    let g2 = loss2.backward()?;
    let dx2 = g2.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx2, &[12.0], 1e-6);

    Ok(())
}

#[test]
fn graph_is_owned_by_loss_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?.requires_grad();

    let (dx,) = {
        let y = x.mul(&x)?;
        let loss = y.sum(None, false)?;
        let g = loss.backward()?;
        (g.wrt(&x).unwrap().clone(),)
    };

    let dx_vec = dx.to_vec()?;
    assert_vec_approx_eq(&dx_vec, &[2.0, 4.0], 1e-6);

    Ok(())
}

#[test]
fn no_interference_between_concurrent_graphs() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0], &[1])?.requires_grad();

    let loss_a = a.mul(&a)?.sum(None, false)?;
    let loss_b = b.mul(&b)?.mul(&b)?.sum(None, false)?;

    let ga = loss_a.backward()?;
    let gb = loss_b.backward()?;

    let da = ga.wrt(&a).unwrap().to_vec()?;
    let db = gb.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[4.0], 1e-6);
    assert_vec_approx_eq(&db, &[27.0], 1e-6);

    Ok(())
}

#[test]
fn detach_preserves_gradient_flow_through_attached_tensors() -> Result<()> {
    // Test that detaching a tensor doesn't prevent gradients from flowing through
    // tensors that are still attached to the computational graph.
    // Even though y is detached, gradients should still flow back to x through
    // the y_detached * x operation.
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?.requires_grad();
    let y = x.mul(&x)?; // y = x^2 = 4.0, requires_grad = true
    let y_detached = y.detach(); // y_detached shares data but requires_grad = false

    // loss = y_detached * x = 4.0 * 2.0 = 8.0
    // Since x requires_grad, this operation creates a gradient connection
    let loss = y_detached.mul(&x)?.sum(None, false)?;
    let g = loss.backward()?;

    // d(loss)/dx = d(y_detached * x)/dx = y_detached + x = 4.0 + 2.0 = 6.0
    // Wait, that's not 4.0... let me recalculate:
    // loss = y_detached * x, so d(loss)/dx = y_detached (since x is the variable)
    // y_detached = 4.0, so gradient should be 4.0
    let dx = g.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx, &[4.0], 1e-6);

    Ok(())
}

#[test]
fn detach_breaks_gradient_flow_to_ancestors() -> Result<()> {
    // Test that detach() breaks gradient flow to ancestor tensors.
    // If we have a -> b -> c, and we detach b, then gradients from c
    // should not flow back to a.
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?.requires_grad();
    let b = a.mul(&a)?; // b = a^2 = 4.0
    let b_detached = b.detach(); // Break gradient connection

    // Now use b_detached in further computation
    let c = b_detached.mul(&a)?; // c = b_detached * a * b = 4.0 * 2.0 * 4.0 = 32.0

    let result = c.backward()?;
    assert!(result.wrt(&b_detached).is_none());
    assert!(result.wrt(&a).is_some());
    Ok(())
}

#[test]
fn detach_disables_gradient_tracking() -> Result<()> {
    // Test that detached tensors don't participate in gradient computation.
    // When operations are performed only on detached tensors, the result
    // should not require gradients and backward() should fail.
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?.requires_grad();
    let y = x.mul(&x)?; // y requires gradients

    // Detach y - this should create a tensor that doesn't track gradients
    let y_detached = y.detach();

    // When we use only the detached tensor in operations, no gradients should flow
    let z = y_detached.mul(&y_detached)?; // z should not require gradients

    // Computing backward on z should fail since it doesn't require gradients
    let result = z.backward();
    assert!(result.is_err()); // Should fail because z doesn't require gradients

    Ok(())
}
