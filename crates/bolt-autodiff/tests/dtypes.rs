use std::any::TypeId;
use std::sync::Arc;

use bolt_autodiff::{Float, Graph, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

fn approx_eq<D: Float>(a: D, b: D, eps: f64) -> bool {
    (a - b).abs().to_f64().unwrap() < eps
}

fn assert_vec_approx_eq<D: Float + std::fmt::Debug>(actual: &[D], expected: &[D], eps: f64) {
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
            "element {} differs: got {:?}, expected {:?} (eps={})",
            i,
            a,
            e,
            eps
        );
    }
}

#[test]
fn test_f32_autodiff_gradients() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f32>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?;
    let b_data = Tensor::from_slice(&backend, &[4.0_f32, 5.0, 6.0], &[3])?;

    let a = graph.leaf(&a_data);
    let b = graph.leaf(&b_data);
    let loss = a.add(&b)?.sum(None)?;

    let grads = graph.backward(&loss)?;
    let expected = vec![1.0_f32; 3];

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &expected, 1e-6);
    assert_vec_approx_eq(&db, &expected, 1e-6);

    Ok(())
}

#[test]
fn test_f64_autodiff_gradients() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let graph = Graph::<CpuBackend, f64>::new(backend.clone());

    let a_data = Tensor::from_slice(&backend, &[1.0_f64, -2.5, 3.25], &[3])?;
    let b_data = Tensor::from_slice(&backend, &[0.5_f64, 4.75, -1.25], &[3])?;

    let a = graph.leaf(&a_data);
    let b = graph.leaf(&b_data);
    let loss = a.add(&b)?.sum(None)?;

    let grads = graph.backward(&loss)?;
    let expected = vec![1.0_f64; 3];

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &expected, 1e-12);
    assert_vec_approx_eq(&db, &expected, 1e-12);

    Ok(())
}

#[test]
fn test_i32_autodiff_is_not_supported() {
    let supported = [TypeId::of::<f32>(), TypeId::of::<f64>()];
    assert!(
        !supported.contains(&TypeId::of::<i32>()),
        "autodiff is intended for floating dtypes"
    );
}
