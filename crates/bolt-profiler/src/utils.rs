use bolt_core::allocator::AllocatorSnapshot;

use crate::report::{DeviceMemoryStats, HostMemoryStats};
use crate::host_mem::HostMemStats;

pub(crate) fn snapshot_delta(after: &AllocatorSnapshot, before: &AllocatorSnapshot) -> AllocatorSnapshot {
    AllocatorSnapshot {
        bytes_requested: after.bytes_requested.saturating_sub(before.bytes_requested),
        bytes_granted: after.bytes_granted.saturating_sub(before.bytes_granted),
        alloc_count: after.alloc_count.saturating_sub(before.alloc_count),
        dealloc_count: after.dealloc_count.saturating_sub(before.dealloc_count),
        peak_in_scope: after.peak_in_scope.saturating_sub(before.peak_in_scope),
        persistent_peak: after.persistent_peak,
        fragmentation_pct: after.fragmentation_pct,
        scratch_bytes: after.scratch_bytes,
        extensions: after.extensions.clone(),
    }
}

pub(crate) fn build_device_memory_stats(base: &AllocatorSnapshot, available: bool) -> DeviceMemoryStats {
    DeviceMemoryStats {
        bytes_requested: base.bytes_requested,
        bytes_granted: base.bytes_granted,
        alloc_count: base.alloc_count,
        dealloc_count: base.dealloc_count,
        peak_in_scope: base.peak_in_scope,
        persistent_peak: base.persistent_peak,
        fragmentation_pct: base.fragmentation_pct,
        scratch_bytes: base.scratch_bytes,
        extensions: base.extensions.clone(),
        available,
    }
}

pub(crate) fn build_host_memory_stats(
    start: &HostMemStats,
    end: &HostMemStats,
    per_interval_peak: usize,
) -> HostMemoryStats {
    let bytes_requested = end
        .cumulative_allocated_bytes
        .saturating_sub(start.cumulative_allocated_bytes) as u64;
    let alloc_count = end.alloc_count.saturating_sub(start.alloc_count) as u64;
    let dealloc_count = end.dealloc_count.saturating_sub(start.dealloc_count) as u64;

    HostMemoryStats {
        bytes_requested,
        bytes_granted: bytes_requested,
        alloc_count,
        dealloc_count,
        peak_in_scope: per_interval_peak as u64,
        persistent_peak: end.peak_allocated_bytes as u64,
        available: true,
    }
}

