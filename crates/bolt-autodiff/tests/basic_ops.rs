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
fn test_add_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?;
    let b_data = Tensor::from_slice(&backend, &[4.0_f32, 5.0, 6.0], &[3])?;

    let a = graph.param(&a_data);
    let b = graph.param(&b_data);
    let c = a.add(&b)?;
    let loss = c.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_mul_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let b_data = Tensor::from_slice(&backend, &[4.0_f32, 5.0], &[2])?;

    let a = graph.param(&a_data);
    let b = graph.param(&b_data);
    let c = a.mul(&b)?;
    let loss = c.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&db, &[2.0, 3.0], 1e-6);

    Ok(())
}

#[test]
fn test_sub_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[5.0_f32, 6.0], &[2])?;
    let b_data = Tensor::from_slice(&backend, &[2.0_f32, 1.0], &[2])?;

    let a = graph.param(&a_data);
    let b = graph.param(&b_data);
    let c = a.sub(&b)?;
    let loss = c.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[-1.0, -1.0], 1e-6);

    Ok(())
}

#[test]
fn test_chain_rule() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let x = graph.param(&x_data);

    let y = x.mul(&x)?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 6.0], 1e-6);

    Ok(())
}

#[test]
fn test_multiple_use_accumulation() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0], &[2])?;
    let x = graph.param(&x_data);

    let y = x.add(&x)?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[2.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_mean_grad() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0, 4.0], &[4])?;
    let x = graph.param(&x_data);

    let loss = x.mean(None, false)?;

    let grads = graph.backward(&loss)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[0.25, 0.25, 0.25, 0.25], 1e-6);

    Ok(())
}

#[test]
fn test_no_grad_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0], &[2])?;
    let y_data = Tensor::from_slice(&backend, &[3.0_f32, 4.0], &[2])?;

    let x = graph.input(&x_data);
    let y = graph.param(&y_data);

    let z = x.add(&y)?;
    let loss = z.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    assert!(grads.wrt(&x).is_none());
    assert!(grads.wrt(&y).is_some());

    let dy = grads.wrt(&y).expect("gradient for y").to_vec()?;
    assert_vec_approx_eq(&dy, &[1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_detach() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[2.0_f32, 3.0], &[2])?;
    let w_data = Tensor::from_slice(&backend, &[1.0_f32, 1.0], &[2])?;

    let x = graph.param(&x_data);
    let w = graph.param(&w_data);

    let y = x.mul(&x)?;
    let y_detached = y.detach()?;
    let z = y_detached.add(&w)?;
    let loss = z.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    assert!(grads.wrt(&x).is_none());
    assert!(grads.wrt(&w).is_some());

    Ok(())
}

#[test]
fn test_reshape_backward() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let x = graph.param(&x_data);

    let y = x.reshape(&[4])?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_transpose_backward() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let x = graph.param(&x_data);

    let y = x.transpose(0, 1)?;
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_graph_clear_invalidates_handles() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32], &[1])?;
    let x = graph.param(&x_data);
    let handle_before = x.handle();

    graph.clear();

    let y_data = Tensor::from_slice(&backend, &[2.0_f32], &[1])?;
    let y = graph.param(&y_data);
    let loss = y.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    assert!(grads.get(&handle_before).is_none());
    assert!(grads.wrt(&y).is_some());

    Ok(())
}

#[test]
fn test_no_grad_guard() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0], &[2])?;
    let x = graph.param(&x_data);

    let y = {
        let _guard = graph.no_grad();
        x.add(&x)?
    };

    assert!(!y.requires_grad()?);

    Ok(())
}

#[test]
fn test_complex_expression() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0], &[2])?;
    let y_data = Tensor::from_slice(&backend, &[3.0_f32, 4.0], &[2])?;

    let x = graph.param(&x_data);
    let y = graph.param(&y_data);

    let a = x.mul(&y)?;
    let b = a.add(&x)?;
    let loss = b.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;
    let dy = grads.wrt(&y).expect("gradient for y").to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&dy, &[1.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_dynamic_control_flow_branch_true() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[3.0_f32], &[1])?;
    let w1_data = Tensor::from_slice(&backend, &[2.0_f32], &[1])?;
    let w2_data = Tensor::from_slice(&backend, &[5.0_f32], &[1])?;

    let x = graph.param(&x_data);
    let w1 = graph.param(&w1_data);
    let w2 = graph.param(&w2_data);

    let condition_value = x.tensor()?.item()?;

    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = graph.backward(&loss)?;

    let dw1 = grads.wrt(&w1).expect("gradient for w1").to_vec()?;
    assert_vec_approx_eq(&dw1, &[3.0], 1e-6);

    assert!(grads.wrt(&w2).is_none());

    Ok(())
}

#[test]
fn test_dynamic_control_flow_branch_false() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[1.0_f32], &[1])?;
    let w1_data = Tensor::from_slice(&backend, &[2.0_f32], &[1])?;
    let w2_data = Tensor::from_slice(&backend, &[5.0_f32], &[1])?;

    let x = graph.param(&x_data);
    let w1 = graph.param(&w1_data);
    let w2 = graph.param(&w2_data);

    let condition_value = x.tensor()?.item()?;

    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = graph.backward(&loss)?;

    let dw2 = grads.wrt(&w2).expect("gradient for w2").to_vec()?;
    assert_vec_approx_eq(&dw2, &[1.0], 1e-6);

    assert!(grads.wrt(&w1).is_none());

    Ok(())
}

#[test]
fn test_dynamic_loop_iterations() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(&backend, &[2.0_f32], &[1])?;
    let w_data = Tensor::from_slice(&backend, &[3.0_f32], &[1])?;

    let x = graph.param(&x_data);
    let w = graph.param(&w_data);

    let iterations = x.tensor()?.item()? as usize;

    let mut result = w.mul(&w)?;
    for _ in 1..iterations {
        result = result.mul(&w)?;
    }
    let loss = result.sum(None, false)?;

    let grads = graph.backward(&loss)?;

    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[27.0], 1e-6);

    Ok(())
}

#[test]
fn test_sum_multi_axis_backward() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(
        &backend,
        &(1..=24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    )?;
    let x = graph.param(&x_data);

    let y = x.sum(Some(&[0, 2]), false)?;

    let grads = graph.backward(&y)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    let expected_grad = vec![1.0_f32; 24];
    assert_vec_approx_eq(&dx, &expected_grad, 1e-6);

    Ok(())
}

#[test]
fn test_mean_multi_axis_backward() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let x_data = Tensor::from_slice(
        &backend,
        &(1..=24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    )?;
    let x = graph.param(&x_data);

    let y = x.mean(Some(&[0, 2]), false)?;

    let grads = graph.backward(&y)?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    let expected_grad = vec![0.125_f32; 24];
    assert_vec_approx_eq(&dx, &expected_grad, 1e-6);

    Ok(())
}
