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
fn dynamic_control_flow_branch_true_records_only_executed_branch() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0_f32], &[1])?.requires_grad();
    let w1 = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0_f32], &[1])?.requires_grad();
    let w2 = Tensor::<CpuBackend, f32>::from_slice(&backend, &[5.0_f32], &[1])?.requires_grad();

    let condition_value = x.item()?;
    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = loss.backward()?;
    let dw1 = grads.wrt(&w1).unwrap().to_vec()?;
    assert_vec_approx_eq(&dw1, &[3.0], 1e-6);
    assert!(grads.wrt(&w2).is_none());

    Ok(())
}

#[test]
fn dynamic_control_flow_branch_false_records_only_executed_branch() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32], &[1])?.requires_grad();
    let w1 = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0_f32], &[1])?.requires_grad();
    let w2 = Tensor::<CpuBackend, f32>::from_slice(&backend, &[5.0_f32], &[1])?.requires_grad();

    let condition_value = x.item()?;
    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = loss.backward()?;
    let dw2 = grads.wrt(&w2).unwrap().to_vec()?;
    assert_vec_approx_eq(&dw2, &[1.0], 1e-6);
    assert!(grads.wrt(&w1).is_none());

    Ok(())
}

#[test]
fn dynamic_loop_iterations_record_full_chain() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0_f32], &[1])?.requires_grad();
    let w = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0_f32], &[1])?.requires_grad();

    let iterations = x.item()? as usize;

    let mut result = w.mul(&w)?;
    for _ in 1..iterations {
        result = result.mul(&w)?;
    }
    let loss = result.sum(None, false)?;

    let grads = loss.backward()?;
    let dw = grads.wrt(&w).unwrap().to_vec()?;
    assert_vec_approx_eq(&dw, &[27.0], 1e-6);

    Ok(())
}
