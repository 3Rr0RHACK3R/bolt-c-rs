use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

// ============================================================================
// mul_scalar tests
// ============================================================================

#[test]
fn mul_scalar_f32() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let result = t.mul_scalar(2.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn mul_scalar_f64() {
    let backend = backend();
    let t = Tensor::<_, f64>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.mul_scalar(0.5).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![0.5, 1.0, 1.5]);
}

#[test]
fn mul_scalar_i32() {
    let backend = backend();
    let t = Tensor::<_, i32>::from_slice(&backend, &[1, 2, 3, 4], &[4]).unwrap();
    let result = t.mul_scalar(3).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![3, 6, 9, 12]);
}

#[test]
fn mul_scalar_by_zero() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.mul_scalar(0.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn mul_scalar_by_one() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.mul_scalar(1.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn mul_scalar_negative() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, -2.0, 3.0], &[3]).unwrap();
    let result = t.mul_scalar(-2.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![-2.0, 4.0, -6.0]);
}

// ============================================================================
// add_scalar tests
// ============================================================================

#[test]
fn add_scalar_f32() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let result = t.add_scalar(10.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn add_scalar_f64() {
    let backend = backend();
    let t = Tensor::<_, f64>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.add_scalar(-0.5).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![0.5, 1.5, 2.5]);
}

#[test]
fn add_scalar_i32() {
    let backend = backend();
    let t = Tensor::<_, i32>::from_slice(&backend, &[1, 2, 3, 4], &[4]).unwrap();
    let result = t.add_scalar(100).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![101, 102, 103, 104]);
}

#[test]
fn add_scalar_zero() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.add_scalar(0.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ============================================================================
// sub_scalar tests
// ============================================================================

#[test]
fn sub_scalar_f32() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let result = t.sub_scalar(5.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![5.0, 15.0, 25.0, 35.0]);
}

#[test]
fn sub_scalar_f64() {
    let backend = backend();
    let t = Tensor::<_, f64>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.sub_scalar(0.5).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![0.5, 1.5, 2.5]);
}

#[test]
fn sub_scalar_i32() {
    let backend = backend();
    let t = Tensor::<_, i32>::from_slice(&backend, &[10, 20, 30, 40], &[4]).unwrap();
    let result = t.sub_scalar(5).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![5, 15, 25, 35]);
}

#[test]
fn sub_scalar_negative() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.sub_scalar(-1.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![2.0, 3.0, 4.0]);
}

// ============================================================================
// div_scalar tests
// ============================================================================

#[test]
fn div_scalar_f32() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let result = t.div_scalar(10.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn div_scalar_f64() {
    let backend = backend();
    let t = Tensor::<_, f64>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.div_scalar(2.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![0.5, 1.0, 1.5]);
}

#[test]
fn div_scalar_i32() {
    let backend = backend();
    let t = Tensor::<_, i32>::from_slice(&backend, &[10, 20, 30, 40], &[4]).unwrap();
    let result = t.div_scalar(5).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![2, 4, 6, 8]);
}

#[test]
fn div_scalar_by_zero_i32() {
    let backend = backend();
    let t = Tensor::<_, i32>::from_slice(&backend, &[10, 20, 30], &[3]).unwrap();
    let result = t.div_scalar(0);
    assert!(result.is_err());
}

#[test]
fn div_scalar_by_zero_f32_produces_inf() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();
    let result = t.div_scalar(0.0).unwrap();
    let values = result.to_vec().unwrap();
    assert!(values.iter().all(|&v| v.is_infinite()));
}

// ============================================================================
// Non-contiguous layout tests
// ============================================================================

#[test]
fn mul_scalar_non_contiguous() {
    let backend = backend();
    let t =
        Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // Transpose to make it non-contiguous
    let t_transposed = t.transpose(0, 1).unwrap();
    let result = t_transposed.mul_scalar(2.0).unwrap();
    // Transposed shape is [3, 2], values should be [[1,4], [2,5], [3,6]] * 2
    assert_eq!(
        result.to_vec().unwrap(),
        vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
    );
}

#[test]
fn add_scalar_non_contiguous() {
    let backend = backend();
    let t =
        Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t_transposed = t.transpose(0, 1).unwrap();
    let result = t_transposed.add_scalar(10.0).unwrap();
    assert_eq!(
        result.to_vec().unwrap(),
        vec![11.0, 14.0, 12.0, 15.0, 13.0, 16.0]
    );
}

#[test]
fn sub_scalar_non_contiguous() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3])
        .unwrap();
    let t_transposed = t.transpose(0, 1).unwrap();
    let result = t_transposed.sub_scalar(5.0).unwrap();
    assert_eq!(
        result.to_vec().unwrap(),
        vec![5.0, 35.0, 15.0, 45.0, 25.0, 55.0]
    );
}

#[test]
fn div_scalar_non_contiguous() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3])
        .unwrap();
    let t_transposed = t.transpose(0, 1).unwrap();
    let result = t_transposed.div_scalar(10.0).unwrap();
    assert_eq!(result.to_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// ============================================================================
// Shape preservation tests
// ============================================================================

#[test]
fn scalar_ops_preserve_shape() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0; 24], &[2, 3, 4]).unwrap();

    let result_mul = t.mul_scalar(2.0).unwrap();
    let result_add = t.add_scalar(1.0).unwrap();
    let result_sub = t.sub_scalar(0.5).unwrap();
    let result_div = t.div_scalar(4.0).unwrap();

    assert_eq!(result_mul.shape().as_slice(), &[2, 3, 4]);
    assert_eq!(result_add.shape().as_slice(), &[2, 3, 4]);
    assert_eq!(result_sub.shape().as_slice(), &[2, 3, 4]);
    assert_eq!(result_div.shape().as_slice(), &[2, 3, 4]);
}

// ============================================================================
// Chaining tests
// ============================================================================

#[test]
fn chained_scalar_ops() {
    let backend = backend();
    let t = Tensor::<_, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

    // (x * 2 + 10 - 5) / 2 = (x * 2 + 5) / 2
    let result = t
        .mul_scalar(2.0)
        .unwrap()
        .add_scalar(10.0)
        .unwrap()
        .sub_scalar(5.0)
        .unwrap()
        .div_scalar(2.0)
        .unwrap();

    // [1, 2, 3, 4] -> [2, 4, 6, 8] -> [12, 14, 16, 18] -> [7, 9, 11, 13] -> [3.5, 4.5, 5.5, 6.5]
    assert_eq!(result.to_vec().unwrap(), vec![3.5, 4.5, 5.5, 6.5]);
}
