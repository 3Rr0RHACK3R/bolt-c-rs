pub mod allocator;
pub mod os_stats;

use std::time::{Duration, Instant};
pub use allocator::{AllocatorStats, TrackingAllocator};
pub use os_stats::{get_os_stats, OsStats};

/// Comprehensive report of a profiled block.
#[derive(Debug, Clone, Copy)]
pub struct ProfileReport {
    pub wall_time: Duration,
    pub cpu_stats: CpuStats,
    pub memory_stats: MemoryStats,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuStats {
    pub user_time: Duration,
    pub sys_time: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Net change in allocated bytes via the Global Allocator (End - Start).
    pub net_allocated_bytes: isize,
    /// Total bytes requested during the block (monotonic).
    pub total_allocated_bytes: usize,
    /// Number of allocation calls made.
    pub alloc_count: usize,
    /// Number of deallocation calls made.
    pub dealloc_count: usize,
    /// Peak physical memory (RSS) used by the process *at the end* of the block.
    pub peak_rss_bytes: u64,
    /// Peak bytes allocated via the Global Allocator *globally* (High water mark).
    pub peak_allocated_bytes: usize,
}

/// Profiles a closure, returning the result and a performance report.
///
/// This function captures:
/// - Wall clock time.
/// - CPU time (User & System).
/// - Memory allocations (if `TrackingAllocator` is configured).
/// - OS-level Memory Usage (RSS).
///
/// # Arguments
///
/// * `allocator` - Optional reference to the global `TrackingAllocator`. 
///                 If `None`, allocator stats will be zeroed.
/// * `f` - The closure to execute.
///
/// # Example
///
/// ```rust,ignore
/// use bolt_profiler::{profile, TrackingAllocator};
///
/// #[global_allocator]
/// static GLOBAL: TrackingAllocator = TrackingAllocator::new();
///
/// let (result, report) = profile(Some(&GLOBAL), || {
///     // computation
///     vec![0u8; 1024]
/// });
/// ```
pub fn profile<F, R>(allocator: Option<&TrackingAllocator>, f: F) -> (R, ProfileReport)
where
    F: FnOnce() -> R,
{
    // 1. Snapshot Start
    let start_time = Instant::now();
    let start_os = get_os_stats();
    let start_alloc = allocator.map(|a| a.stats()).unwrap_or_default();

    // 2. Run Code
    let result = f();

    // 3. Snapshot End
    let end_alloc = allocator.map(|a| a.stats()).unwrap_or_default();
    let end_os = get_os_stats();
    let end_time = Instant::now();

    // 4. Calculate Diff
    let wall_time = end_time.duration_since(start_time);
    
    let cpu_stats = CpuStats {
        user_time: end_os.user_cpu_time.saturating_sub(start_os.user_cpu_time),
        sys_time: end_os.sys_cpu_time.saturating_sub(start_os.sys_cpu_time),
    };

    let memory_stats = MemoryStats {
        net_allocated_bytes: (end_alloc.allocated_bytes as isize) - (start_alloc.allocated_bytes as isize),
        total_allocated_bytes: end_alloc.cumulative_allocated_bytes.saturating_sub(start_alloc.cumulative_allocated_bytes),
        alloc_count: end_alloc.alloc_count.saturating_sub(start_alloc.alloc_count),
        dealloc_count: end_alloc.dealloc_count.saturating_sub(start_alloc.dealloc_count),
        peak_rss_bytes: end_os.rss_bytes,
        peak_allocated_bytes: end_alloc.peak_allocated_bytes,
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
