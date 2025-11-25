use std::time::Duration;

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    pub net_allocated_bytes: isize,
    pub total_allocated_bytes: usize,
    pub alloc_count: usize,
    pub dealloc_count: usize,
    pub peak_rss_bytes: u64,
    pub scope_peak_bytes: usize,
}
