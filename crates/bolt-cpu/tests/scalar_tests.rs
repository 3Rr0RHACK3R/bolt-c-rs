use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;
use std::sync::Arc;

#[test]
fn test_scalar_creation() {
    let backend = Arc::new(CpuBackend::new());
    let data = [42.0];
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[])
        .expect("failed to create scalar tensor");
    assert_eq!(tensor.shape().as_slice(), &[]);
    assert_eq!(tensor.numel(), 1);
    assert_eq!(tensor.rank(), 0);
    assert_eq!(tensor.item().unwrap(), 42.0);
}

#[test]
fn test_scalar_arithmetic() {
    let backend = Arc::new(CpuBackend::new());
    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[]).unwrap();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0], &[]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape().as_slice(), &[]);
    assert_eq!(c.item().unwrap(), 5.0);
}

#[test]
fn test_scalar_broadcast() {
    let backend = Arc::new(CpuBackend::new());
    let scalar = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[]).unwrap();
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3]).unwrap();

    // scalar + vector -> vector
    let result = scalar.add(&tensor).unwrap();
    assert_eq!(result.shape().as_slice(), &[3]);
    let vals = result.to_vec().unwrap();
    assert_eq!(vals, vec![3.0, 4.0, 5.0]);
}
