use std::time::Duration;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub wall_time: Duration,
    pub cpu_stats: CpuStats,
    pub memory_stats: MemoryStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuStats {
    pub user_time: Duration,
    pub sys_time: Duration,
    pub thread_time: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryStatsSource {
    #[default]
    Unavailable,
    BackendAllocator,
    TrackingAllocator,
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Where these stats came from (backend allocator, tracking fallback, or unavailable).
    pub source: MemoryStatsSource,
    /// Bytes requested by the caller before any allocator alignment.
    pub bytes_requested: u64,
    /// Bytes actually granted by the allocator (post-alignment).
    pub bytes_granted: u64,
    /// Number of allocations performed.
    pub alloc_count: u64,
    /// Number of deallocations performed.
    pub dealloc_count: u64,
    /// Peak live bytes observed within the last scope.
    pub peak_in_scope: u64,
    /// Global high-water mark across the allocator lifetime.
    pub persistent_peak: u64,
    /// Reported external fragmentation percentage if supported.
    pub fragmentation_pct: Option<f32>,
    /// Persistent scratch/workspace bytes held by the allocator.
    pub scratch_bytes: Option<u64>,
    /// Backend-specific extension metrics.
    pub extensions: HashMap<&'static str, i64>,
    /// Process RSS peak sampled alongside the report.
    pub peak_rss_bytes: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            source: MemoryStatsSource::Unavailable,
            bytes_requested: 0,
            bytes_granted: 0,
            alloc_count: 0,
            dealloc_count: 0,
            peak_in_scope: 0,
            persistent_peak: 0,
            fragmentation_pct: None,
            scratch_bytes: None,
            extensions: HashMap::new(),
            peak_rss_bytes: 0,
        }
    }
}
