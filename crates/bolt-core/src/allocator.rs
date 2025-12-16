use std::collections::HashMap;

use bitflags::bitflags;

use crate::{
    dtype::{DType, NativeType},
    error::{Error, Result},
};

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DiagnosticsCaps: u32 {
        /// Supports scoped begin/end tracking.
        const SUPPORTS_SCOPE = 1 << 0;
        /// Reports fragmentation percentage.
        const SUPPORTS_FRAGMENTATION = 1 << 1;
        /// Reports persistent scratch/workspace bytes.
        const SUPPORTS_SCRATCH = 1 << 2;
        /// Emits backend-specific extension metrics.
        const SUPPORTS_EXTENSIONS = 1 << 3;
    }
}

#[derive(Debug, Clone, Default)]
pub struct AllocatorSnapshot {
    /// Total bytes requested by callers before allocator alignment.
    pub bytes_requested: u64,
    /// Bytes granted after allocator alignment or rounding.
    pub bytes_granted: u64,
    /// Number of allocations performed.
    pub alloc_count: u64,
    /// Number of deallocations performed.
    pub dealloc_count: u64,
    /// Peak live bytes observed within the last scope.
    pub peak_in_scope: u64,
    /// Global high-water mark across allocator lifetime.
    pub persistent_peak: u64,
    /// External fragmentation percentage if reported.
    pub fragmentation_pct: Option<f32>,
    /// Persistent scratch/workspace pool size.
    pub scratch_bytes: Option<u64>,
    /// Backend-specific extension metrics.
    pub extensions: HashMap<&'static str, i64>,
}

/// Aggregate allocator diagnostics. Implementations should report global totals;
/// overlapping scopes are allowed and should not corrupt aggregates.
pub trait AllocatorDiagnostics: Send + Sync + 'static {
    // TODO: extend diagnostics to cover backend kernel/workspace stats for GPU backends.
    fn capabilities(&self) -> DiagnosticsCaps {
        DiagnosticsCaps::empty()
    }

    fn snapshot(&self) -> AllocatorSnapshot {
        AllocatorSnapshot::default()
    }

    fn begin_scope(&self) {}

    /// Returns per-scope deltas when supported; otherwise None.
    fn end_scope(&self) -> Option<AllocatorSnapshot> {
        None
    }
}

pub trait StorageAllocator<D: NativeType>:
    AllocatorDiagnostics + Clone + Send + Sync + 'static
{
    type Storage: Clone + Send + Sync + 'static;

    fn allocate(&self, len: usize) -> Result<Self::Storage>;
    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage>;

    fn allocate_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let elem_size = dtype.size_in_bytes();
        if !len_bytes.is_multiple_of(elem_size) {
            return Err(Error::invalid_shape(format!(
                "byte length {} is not aligned to dtype size {}",
                len_bytes, elem_size
            )));
        }
        self.allocate(len_bytes / elem_size)
    }

    fn allocate_zeroed_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let elem_size = dtype.size_in_bytes();
        if !len_bytes.is_multiple_of(elem_size) {
            return Err(Error::invalid_shape(format!(
                "byte length {} is not aligned to dtype size {}",
                len_bytes, elem_size
            )));
        }
        self.allocate_zeroed(len_bytes / elem_size)
    }

    fn release(&self, _storage: Self::Storage) {}
}
