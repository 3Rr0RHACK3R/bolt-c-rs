use std::time::Duration;
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct DeviceTimeStats {
    pub kernel_time: Option<Duration>,
    pub queue_time: Option<Duration>,
    pub extensions: HashMap<&'static str, i64>,
    pub available: bool,
}

impl Default for DeviceTimeStats {
    fn default() -> Self {
        Self {
            kernel_time: None,
            queue_time: None,
            extensions: HashMap::new(),
            available: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub host: HostMemoryStats,
    pub device: DeviceMemoryStats,
    pub peak_rss_bytes: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            host: HostMemoryStats::default(),
            device: DeviceMemoryStats::default(),
            peak_rss_bytes: 0,
        }
    }
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

#[derive(Debug, Clone)]
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

impl Default for DeviceMemoryStats {
    fn default() -> Self {
        Self {
            bytes_requested: 0,
            bytes_granted: 0,
            alloc_count: 0,
            dealloc_count: 0,
            peak_in_scope: 0,
            persistent_peak: 0,
            fragmentation_pct: None,
            scratch_bytes: None,
            extensions: HashMap::new(),
            available: false,
        }
    }
}
