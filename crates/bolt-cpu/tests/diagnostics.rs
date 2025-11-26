use bolt_core::allocator::DiagnosticsCaps;
use bolt_core::{AllocatorDiagnostics, Backend};
use bolt_cpu::CpuBackend;

#[test]
fn cpu_allocator_capabilities_follow_feature_flag() {
    let backend = CpuBackend::new();
    let caps = <CpuBackend as Backend<f32>>::allocator(&backend).capabilities();
    if cfg!(feature = "diagnostics") {
        assert!(caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE));
    } else {
        assert!(caps.is_empty());
    }
}
