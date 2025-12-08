use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use bolt_core::AllocatorDiagnostics;
use bolt_core::allocator::{AllocatorSnapshot, DiagnosticsCaps};
use parking_lot::Mutex;

use crate::host_mem::{HostMemStats, HostMemTracker};
use crate::os_stats::{OsStats, get_os_stats};
use crate::registry::{OpCategory, OpId, Registry};
use crate::report::{
    DeviceTimeStats, HostMemoryStats, HostTimeStats, MemoryStats, ProfileReport, TimeStats,
};
use crate::utils::{build_device_memory_stats, build_host_memory_stats, snapshot_delta};

#[derive(Clone, Debug)]
pub struct Profiler {
    registry: Arc<Mutex<Registry>>,
    host_mem: Option<&'static HostMemTracker>,
}

impl Profiler {
    pub fn new(host_mem: Option<&'static HostMemTracker>) -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
            host_mem,
        }
    }

    pub fn registry(&self) -> Arc<Mutex<Registry>> {
        Arc::clone(&self.registry)
    }

    pub fn clear(&self) {
        self.registry.lock().clear();
    }

    pub fn with_scope<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let parent = current_scope();
        let empty_report = ProfileReport {
            time: TimeStats {
                host: HostTimeStats::default(),
                device: DeviceTimeStats::default(),
            },
            memory: MemoryStats::default(),
        };

        let id = self.registry.lock().record(
            name,
            OpCategory::UserScope,
            Vec::new(),
            parent,
            &empty_report,
        );

        push_scope(id);
        let result = f();
        pop_scope();
        result
    }

    pub fn begin_op(&self, allocator: &impl AllocatorDiagnostics) -> ActiveProfile {
        let caps = allocator.capabilities();
        if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            allocator.begin_scope();
        }
        let device_snapshot = allocator.snapshot();

        if let Some(tracker) = self.host_mem {
            tracker.begin_interval();
        }

        ActiveProfile {
            start_time: Instant::now(),
            start_os: get_os_stats(),
            device_snapshot,
            host_mem_start: self.host_mem.map(|h| h.stats()),
        }
    }

    pub fn end_op(
        &self,
        active: ActiveProfile,
        allocator: &impl AllocatorDiagnostics,
        name: &str,
        category: OpCategory,
        shapes: Vec<Vec<usize>>,
    ) {
        let end_time = Instant::now();
        let end_os = get_os_stats();
        let wall_time = end_time.duration_since(active.start_time);

        let host_time = HostTimeStats {
            wall_time,
            user_time: end_os
                .user_cpu_time
                .saturating_sub(active.start_os.user_cpu_time),
            sys_time: end_os
                .sys_cpu_time
                .saturating_sub(active.start_os.sys_cpu_time),
            thread_time: end_os
                .thread_cpu_time
                .saturating_sub(active.start_os.thread_cpu_time),
            available: true,
        };

        let caps = allocator.capabilities();
        let device_memory = if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            let scope_snapshot = allocator.end_scope();
            let after = allocator.snapshot();
            let base =
                scope_snapshot.unwrap_or_else(|| snapshot_delta(&after, &active.device_snapshot));
            build_device_memory_stats(&base, true)
        } else {
            let end_snapshot = allocator.snapshot();
            let base = snapshot_delta(&end_snapshot, &active.device_snapshot);
            let available = !caps.is_empty()
                || base.bytes_requested > 0
                || base.alloc_count > 0
                || base.dealloc_count > 0;
            build_device_memory_stats(&base, available)
        };

        let host_memory = if let Some(tracker) = self.host_mem {
            let per_interval_peak = tracker.end_interval();
            let end_stats = tracker.stats();
            let start_stats = active.host_mem_start.unwrap_or_default();
            build_host_memory_stats(&start_stats, &end_stats, per_interval_peak)
        } else {
            HostMemoryStats::default()
        };

        let report = ProfileReport {
            time: TimeStats {
                host: host_time,
                device: DeviceTimeStats::default(),
            },
            memory: MemoryStats {
                host: host_memory,
                device: device_memory,
                peak_rss_bytes: end_os.rss_bytes,
            },
        };

        self.registry
            .lock()
            .record(name, category, shapes, current_scope(), &report);
    }
}

pub struct ActiveProfile {
    start_time: Instant,
    start_os: OsStats,
    device_snapshot: AllocatorSnapshot,
    host_mem_start: Option<HostMemStats>,
}

thread_local! {
    static SCOPE_STACK: RefCell<Vec<OpId>> = const { RefCell::new(Vec::new()) };
}

fn current_scope() -> Option<OpId> {
    SCOPE_STACK.with(|s| s.borrow().last().copied())
}

fn push_scope(id: OpId) {
    SCOPE_STACK.with(|s| s.borrow_mut().push(id));
}

fn pop_scope() {
    SCOPE_STACK.with(|s| {
        let _ = s.borrow_mut().pop();
    });
}
