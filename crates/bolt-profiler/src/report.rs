use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub time: TimeStats,
    pub memory: MemoryStats,
    // Future axes:
    // pub compute: ComputeStats,
    // pub io: IoStats,
}

#[derive(Debug, Clone)]
pub struct TimeStats {
    pub host: HostTimeStats,
    pub device: DeviceTimeStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HostTimeStats {
    pub wall_time: Duration,
    pub user_time: Duration,
    pub sys_time: Duration,
    pub thread_time: Duration,
    pub available: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceTimeStats {
    pub kernel_time: Option<Duration>,
    pub queue_time: Option<Duration>,
    pub extensions: HashMap<&'static str, i64>,
    pub available: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub host: HostMemoryStats,
    pub device: DeviceMemoryStats,
    pub peak_rss_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HostMemoryStats {
    pub bytes_requested: u64,
    pub bytes_granted: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
    pub peak_in_scope: u64,
    pub persistent_peak: u64,
    pub available: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceMemoryStats {
    pub bytes_requested: u64,
    pub bytes_granted: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
    pub peak_in_scope: u64,
    pub persistent_peak: u64,
    pub fragmentation_pct: Option<f32>,
    pub scratch_bytes: Option<u64>,
    pub extensions: HashMap<&'static str, i64>,
    pub available: bool,
}
