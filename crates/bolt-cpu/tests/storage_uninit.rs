use std::sync::Arc;

use bolt_core::{
    backend::Backend,
    layout::Layout,
    NativeType,
    shape::ConcreteShape,
};
use bolt_cpu::CpuBackend;

#[cfg(miri)]
#[test]
fn miri_no_ub_on_fill_allocation() {
    let backend = Arc::new(CpuBackend::new());
    let shape = ConcreteShape::from_slice(&[4]).unwrap();
    let layout = Layout::contiguous(shape);

    // This exercised UB previously because fill operated over a logically
    // uninitialized allocation.
    let storage = backend.fill(&layout, 7.0f32).unwrap();

    let mut dst = vec![0.0f32; 4];
    backend.read(&storage, &layout, &mut dst).unwrap();
    assert_eq!(dst, vec![7.0f32; 4]);
}

#[test]
fn fill_and_read_strided_layout() {
    let backend = Arc::new(CpuBackend::new());
    let shape = ConcreteShape::from_slice(&[2, 2]).unwrap();
    // Strided layout with an offset into the underlying buffer.
    let layout =
        Layout::with_strides(shape, &[3, 1], f32::DTYPE.size_in_bytes()).unwrap();

    let storage = backend.fill(&layout, 5.0f32).unwrap();

    // Reading through the same layout should yield the fill value.
    let mut dst = vec![0.0f32; 4];
    backend.read(&storage, &layout, &mut dst).unwrap();
    assert_eq!(dst, vec![5.0f32; 4]);
}
