use bolt_core::Result;
use bolt_core::{AllocatorDiagnostics, StorageAllocator};

#[derive(Clone)]
struct NoopAlloc;

impl StorageAllocator<f32> for NoopAlloc {
    type Storage = ();

    fn allocate(&self, _len: usize) -> Result<Self::Storage> {
        Ok(())
    }

    fn allocate_zeroed(&self, _len: usize) -> Result<Self::Storage> {
        Ok(())
    }
}

impl AllocatorDiagnostics for NoopAlloc {}

#[test]
fn default_diagnostics_are_noop() {
    let alloc = NoopAlloc;
    assert!(alloc.capabilities().is_empty());

    let snapshot = alloc.snapshot();
    assert_eq!(snapshot.bytes_requested, 0);
    assert_eq!(snapshot.bytes_granted, 0);
    assert_eq!(snapshot.alloc_count, 0);
    assert_eq!(snapshot.dealloc_count, 0);
    assert_eq!(snapshot.peak_in_scope, 0);
    assert_eq!(snapshot.persistent_peak, 0);
    assert!(snapshot.extensions.is_empty());

    alloc.begin_scope();
    assert!(alloc.end_scope().is_none());
}
