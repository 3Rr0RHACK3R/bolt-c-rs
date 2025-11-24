use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use std::sync::Arc;

#[test]
fn display_and_debug_render_scalar_without_panic() {
    let backend = Arc::new(CpuBackend::new());
    let scalar = Tensor::from_slice(&backend, &[42.0f32], &[]).expect("scalar construction");

    let display = format!("{scalar}");
    assert_eq!(display, "tensor(42, shape=[], dtype=f32, device=cpu)");

    let debug = format!("{scalar:?}");
    assert_eq!(debug, display);
}
