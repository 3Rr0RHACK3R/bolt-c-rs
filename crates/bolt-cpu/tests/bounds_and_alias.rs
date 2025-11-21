use std::sync::Arc;

use bolt_core::shape::ConcreteShape;
use bolt_core::{Backend, Error, Layout, Tensor};
use bolt_cpu::CpuBackend;

#[test]
fn read_rejects_out_of_bounds_layout() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::zeros(&backend, &[2, 2]).unwrap();

    let bad_layout =
        Layout::with_strides(ConcreteShape::from_slice(&[2, 2]).unwrap(), &[4, 1], 0).unwrap();
    let mut dst = vec![0f32; 4];
    let err = backend
        .read(tensor.storage(), &bad_layout, &mut dst)
        .expect_err("expected bounds error");
    assert!(matches!(err, Error::InvalidShape { .. }));
}

#[test]
fn write_rejects_shared_storage() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::zeros(&backend, &[2, 2]).unwrap();
    let mut shared = tensor.storage().clone();
    let err = backend
        .write(&mut shared, tensor.layout(), &[0f32; 4])
        .expect_err("expected shared-storage write to fail");
    assert!(matches!(err, Error::OpError(_)));
}
