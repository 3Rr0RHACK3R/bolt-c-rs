use std::{marker::PhantomData, sync::Arc};

#[cfg(feature = "diagnostics")]
use std::{
    cell::RefCell,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use bolt_core::{
    allocator::{AllocatorDiagnostics, AllocatorSnapshot, DiagnosticsCaps, StorageAllocator},
    dtype::{DType, NativeType},
    error::Result,
};

use super::context::CpuContext;

use super::storage::{CpuStorage, StorageBlock, make_cpu_handle};

#[cfg(feature = "diagnostics")]
#[derive(Debug, Default)]
pub(crate) struct CpuAllocTelemetry {
    bytes_requested: AtomicU64,
    bytes_granted: AtomicU64,
    alloc_count: AtomicU64,
    dealloc_count: AtomicU64,
    live_bytes: AtomicU64,
    persistent_peak: AtomicU64,
    scope_peak: AtomicU64,
    scope_depth: AtomicUsize,
}

#[cfg(feature = "diagnostics")]
#[derive(Clone)]
struct ScopeBaseline {
    snapshot: AllocatorSnapshot,
    live_bytes: u64,
}

#[cfg(feature = "diagnostics")]
thread_local! {
    static SCOPE_BASELINES: RefCell<Vec<ScopeBaseline>> = RefCell::new(Vec::new());
}

#[cfg(feature = "diagnostics")]
impl CpuAllocTelemetry {
    fn record_alloc(&self, requested: u64, granted: u64) {
        self.bytes_requested.fetch_add(requested, Ordering::Relaxed);
        self.bytes_granted.fetch_add(granted, Ordering::Relaxed);
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        let prev_live = self.live_bytes.fetch_add(granted, Ordering::Relaxed);
        let current = prev_live + granted;
        self.bump_peak(&self.persistent_peak, current);
        if self.scope_depth.load(Ordering::Relaxed) > 0 {
            self.bump_peak(&self.scope_peak, current);
        }
    }

    pub(crate) fn record_dealloc(&self, bytes: u64) {
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .live_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            });
    }

    fn live_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed)
    }

    fn reset_scope_peak_to_current(&self) {
        let live = self.live_bytes();
        self.scope_peak.store(live, Ordering::Relaxed);
    }

    fn scope_peak_delta(&self, baseline_live: u64) -> u64 {
        self.scope_peak
            .load(Ordering::Relaxed)
            .saturating_sub(baseline_live)
    }

    fn snapshot(&self) -> AllocatorSnapshot {
        let mut snapshot = AllocatorSnapshot::default();
        snapshot.bytes_requested = self.bytes_requested.load(Ordering::Relaxed);
        snapshot.bytes_granted = self.bytes_granted.load(Ordering::Relaxed);
        snapshot.alloc_count = self.alloc_count.load(Ordering::Relaxed);
        snapshot.dealloc_count = self.dealloc_count.load(Ordering::Relaxed);
        snapshot.peak_in_scope = self.scope_peak.load(Ordering::Relaxed);
        snapshot.persistent_peak = self.persistent_peak.load(Ordering::Relaxed);
        snapshot
    }

    fn bump_peak(&self, target: &AtomicU64, value: u64) {
        let mut current = target.load(Ordering::Relaxed);
        while value > current {
            match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuAllocator<D: NativeType> {
    #[allow(dead_code)]
    context: Arc<CpuContext>,
    #[cfg(feature = "diagnostics")]
    diagnostics: Arc<CpuAllocTelemetry>,
    _marker: PhantomData<D>,
}

impl<D: NativeType> CpuAllocator<D> {
    pub fn new(context: Arc<CpuContext>) -> Self {
        #[cfg(feature = "diagnostics")]
        let diagnostics = Arc::new(CpuAllocTelemetry::default());

        Self {
            context,
            #[cfg(feature = "diagnostics")]
            diagnostics,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn with_diagnostics(
        context: Arc<CpuContext>,
        diagnostics: Arc<CpuAllocTelemetry>,
    ) -> Self {
        Self {
            context,
            diagnostics,
            _marker: PhantomData,
        }
    }

    pub fn num_threads(&self) -> usize {
        1
    }
}

impl<D: NativeType> StorageAllocator<D> for CpuAllocator<D> {
    type Storage = CpuStorage<D>;

    fn allocate(&self, len: usize) -> Result<Self::Storage> {
        let handle = make_cpu_handle::<D>(len)?;
        #[cfg(feature = "diagnostics")]
        {
            let bytes = handle.len_bytes() as u64;
            self.diagnostics.record_alloc(bytes, bytes);
        }
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(
                len,
                false,
                #[cfg(feature = "diagnostics")]
                Some(Arc::clone(&self.diagnostics)),
            )),
        ))
    }

    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage> {
        let handle = make_cpu_handle::<D>(len)?;
        #[cfg(feature = "diagnostics")]
        {
            let bytes = handle.len_bytes() as u64;
            self.diagnostics.record_alloc(bytes, bytes);
        }
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(
                len,
                true,
                #[cfg(feature = "diagnostics")]
                Some(Arc::clone(&self.diagnostics)),
            )),
        ))
    }

    fn allocate_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let len = len_bytes / dtype.size_in_bytes();
        let handle = make_cpu_handle::<D>(len)?;
        #[cfg(feature = "diagnostics")]
        {
            let bytes = handle.len_bytes() as u64;
            self.diagnostics.record_alloc(len_bytes as u64, bytes);
        }
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(
                len,
                false,
                #[cfg(feature = "diagnostics")]
                Some(Arc::clone(&self.diagnostics)),
            )),
        ))
    }

    fn allocate_zeroed_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let len = len_bytes / dtype.size_in_bytes();
        let handle = make_cpu_handle::<D>(len)?;
        #[cfg(feature = "diagnostics")]
        {
            let bytes = handle.len_bytes() as u64;
            self.diagnostics.record_alloc(len_bytes as u64, bytes);
        }
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(
                len,
                true,
                #[cfg(feature = "diagnostics")]
                Some(Arc::clone(&self.diagnostics)),
            )),
        ))
    }
}

impl<D: NativeType> AllocatorDiagnostics for CpuAllocator<D> {
    #[cfg(feature = "diagnostics")]
    fn capabilities(&self) -> DiagnosticsCaps {
        DiagnosticsCaps::SUPPORTS_SCOPE
    }

    #[cfg(not(feature = "diagnostics"))]
    fn capabilities(&self) -> DiagnosticsCaps {
        DiagnosticsCaps::empty()
    }

    fn snapshot(&self) -> AllocatorSnapshot {
        #[cfg(feature = "diagnostics")]
        {
            return self.diagnostics.snapshot();
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            return AllocatorSnapshot::default();
        }
    }

    fn begin_scope(&self) {
        #[cfg(feature = "diagnostics")]
        {
            let baseline = ScopeBaseline {
                snapshot: self.diagnostics.snapshot(),
                live_bytes: self.diagnostics.live_bytes(),
            };
            SCOPE_BASELINES.with(|stack| stack.borrow_mut().push(baseline));
            let depth = self.diagnostics.scope_depth.fetch_add(1, Ordering::Relaxed);
            if depth == 0 {
                self.diagnostics.reset_scope_peak_to_current();
            }
        }
    }

    fn end_scope(&self) -> Option<AllocatorSnapshot> {
        #[cfg(feature = "diagnostics")]
        {
            let baseline = match SCOPE_BASELINES.with(|stack| stack.borrow_mut().pop()) {
                Some(b) => b,
                None => {
                    debug_assert!(
                        false,
                        "end_scope called without matching begin_scope; ignoring scope stats"
                    );
                    return None;
                }
            };
            let current = self.diagnostics.snapshot();
            let mut delta = AllocatorSnapshot::default();
            delta.bytes_requested = current
                .bytes_requested
                .saturating_sub(baseline.snapshot.bytes_requested);
            delta.bytes_granted = current
                .bytes_granted
                .saturating_sub(baseline.snapshot.bytes_granted);
            delta.alloc_count = current
                .alloc_count
                .saturating_sub(baseline.snapshot.alloc_count);
            delta.dealloc_count = current
                .dealloc_count
                .saturating_sub(baseline.snapshot.dealloc_count);
            delta.peak_in_scope = self.diagnostics.scope_peak_delta(baseline.live_bytes);
            delta.persistent_peak = current.persistent_peak;
            delta.fragmentation_pct = current.fragmentation_pct;
            delta.scratch_bytes = current.scratch_bytes;
            delta.extensions = current.extensions.clone();

            let prev_depth = self
                .diagnostics
                .scope_depth
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |depth| {
                    depth.checked_sub(1)
                })
                .unwrap_or(0);
            if prev_depth <= 1 {
                self.diagnostics.reset_scope_peak_to_current();
            }

            Some(delta)
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            None
        }
    }
}
