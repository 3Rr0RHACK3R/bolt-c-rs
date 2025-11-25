pub mod allocator;
pub mod backend;
pub mod os_stats;
pub mod report;

use std::time::Instant;

pub use allocator::{AllocatorStats, TrackingAllocator};
pub use backend::{OpStats, ProfiledBackend, Registry};
pub use os_stats::{OsStats, get_os_stats};
pub use report::{CpuStats, MemoryStats, ProfileReport};

pub fn profile<F, R>(allocator: Option<&TrackingAllocator>, f: F) -> (R, ProfileReport)
where
    F: FnOnce() -> R,
{
    let start_time = Instant::now();
    let start_os = get_os_stats();
    let start_alloc = allocator.map(|a| a.stats()).unwrap_or_default();

    if let Some(a) = allocator {
        a.begin_scope();
    }

    let result = f();

    let scope_peak_delta = allocator.map(|a| a.end_scope()).unwrap_or(0);
    let end_alloc = allocator.map(|a| a.stats()).unwrap_or_default();
    let end_os = get_os_stats();
    let end_time = Instant::now();

    let wall_time = end_time.duration_since(start_time);

    let cpu_stats = CpuStats {
        user_time: end_os.user_cpu_time.saturating_sub(start_os.user_cpu_time),
        sys_time: end_os.sys_cpu_time.saturating_sub(start_os.sys_cpu_time),
        thread_time: end_os
            .thread_cpu_time
            .saturating_sub(start_os.thread_cpu_time),
    };

    let memory_stats = MemoryStats {
        net_allocated_bytes: (end_alloc.allocated_bytes as isize)
            - (start_alloc.allocated_bytes as isize),
        total_allocated_bytes: end_alloc
            .cumulative_allocated_bytes
            .saturating_sub(start_alloc.cumulative_allocated_bytes),
        alloc_count: end_alloc
            .alloc_count
            .saturating_sub(start_alloc.alloc_count),
        dealloc_count: end_alloc
            .dealloc_count
            .saturating_sub(start_alloc.dealloc_count),
        peak_rss_bytes: end_os.rss_bytes,
        scope_peak_bytes: scope_peak_delta,
    };

    (
        result,
        ProfileReport {
            wall_time,
            cpu_stats,
            memory_stats,
        },
    )
}
