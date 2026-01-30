use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

// ============================================================================
// mul_scalar gradient tests
// ============================================================================

#[test]
fn mul_scalar_grad_simple() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad();

    let y = x.mul_scalar(2.0).unwrap().sum(None, false).unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x * 2)) = 2 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![2.0, 2.0, 2.0]);
}

#[test]
fn mul_scalar_grad_chain_rule() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad();

    // y = sum((x * 3) * 2) = sum(x * 6)
    let y = x
        .mul_scalar(3.0)
        .unwrap()
        .mul_scalar(2.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x * 6)) = 6 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![6.0, 6.0, 6.0]);
}

#[test]
fn mul_scalar_grad_with_other_ops() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap()
        .requires_grad();

    // y = sum((x + x) * 2) = sum(2x * 2) = sum(4x)
    let y = x
        .add(&x)
        .unwrap()
        .mul_scalar(2.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(4x)) = 4 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![4.0, 4.0, 4.0, 4.0]);
}

// ============================================================================
// add_scalar gradient tests
// ============================================================================

#[test]
fn add_scalar_grad_simple() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad();

    let y = x.add_scalar(10.0).unwrap().sum(None, false).unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x + c)) = 1 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn add_scalar_grad_chain() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad();

    // y = sum((x + 10) + 20) = sum(x + 30)
    let y = x
        .add_scalar(10.0)
        .unwrap()
        .add_scalar(20.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x + 30)) = 1 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

// ============================================================================
// sub_scalar gradient tests
// ============================================================================

#[test]
fn sub_scalar_grad_simple() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0], &[3])
        .unwrap()
        .requires_grad();

    let y = x.sub_scalar(5.0).unwrap().sum(None, false).unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x - c)) = 1 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn sub_scalar_grad_chain() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0], &[3])
        .unwrap()
        .requires_grad();

    // y = sum((x - 5) - 3) = sum(x - 8)
    let y = x
        .sub_scalar(5.0)
        .unwrap()
        .sub_scalar(3.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x - 8)) = 1 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

// ============================================================================
// div_scalar gradient tests
// ============================================================================

#[test]
fn div_scalar_grad_simple() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0], &[3])
        .unwrap()
        .requires_grad();

    let y = x.div_scalar(5.0).unwrap().sum(None, false).unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x / c)) = 1/c for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![0.2, 0.2, 0.2]);
}

#[test]
fn div_scalar_grad_chain() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0], &[3])
        .unwrap()
        .requires_grad();

    // y = sum((x / 2) / 5) = sum(x / 10)
    let y = x
        .div_scalar(2.0)
        .unwrap()
        .div_scalar(5.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x / 10)) = 0.1 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![0.1, 0.1, 0.1]);
}

// ============================================================================
// Combined scalar ops gradient tests
// ============================================================================

#[test]
fn combined_scalar_ops_grad() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4])
        .unwrap()
        .requires_grad();

    // y = sum((x * 2 + 10 - 5) / 2) = sum((2x + 5) / 2) = sum(x + 2.5)
    let y = x
        .mul_scalar(2.0)
        .unwrap()
        .add_scalar(10.0)
        .unwrap()
        .sub_scalar(5.0)
        .unwrap()
        .div_scalar(2.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = y.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();

    // d/dx (sum(x + 2.5)) = 1 for each element
    // But we need to track through the chain:
    // - d/dx (y) where y = sum(z / 2) -> grad_z = 1/2
    // - z = w - 5 -> grad_w = 1/2
    // - w = v + 10 -> grad_v = 1/2
    // - v = x * 2 -> grad_x = 1/2 * 2 = 1
    assert_eq!(grad_x.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn scalar_ops_with_tensor_ops_grad() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap()
        .requires_grad();
    let y = Tensor::<_, f32>::from_slice(&backend, &[1.0, 1.0, 1.0, 1.0], &[2, 2])
        .unwrap()
        .requires_grad();

    // z = sum((x + y) * 3)
    let z = x
        .add(&y)
        .unwrap()
        .mul_scalar(3.0)
        .unwrap()
        .sum(None, false)
        .unwrap();

    let grads = z.backward().unwrap();
    let grad_x = grads.wrt(&x).unwrap();
    let grad_y = grads.wrt(&y).unwrap();

    // d/dx (sum((x + y) * 3)) = 3 for each element
    assert_eq!(grad_x.to_vec().unwrap(), vec![3.0, 3.0, 3.0, 3.0]);
    assert_eq!(grad_y.to_vec().unwrap(), vec![3.0, 3.0, 3.0, 3.0]);
}

// ============================================================================
// No-grad context tests
// ============================================================================

#[test]
fn scalar_ops_no_grad() {
    let backend = backend();
    let x = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();

    // Without requires_grad, operations should still work
    let result = x.mul_scalar(2.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);

    // Should not be able to call backward
    let sum_result = result.sum(None, false).unwrap();
    assert!(sum_result.backward().is_err());
}
