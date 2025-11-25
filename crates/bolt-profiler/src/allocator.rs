use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A Global Allocator that tracks memory usage statistics.
///
/// # Usage
/// To use this allocator, you must register it as the global allocator in your binary's main file
/// (usually `main.rs` or `lib.rs` of the final executable).
///
/// ```rust,ignore
/// use bolt_profiler::TrackingAllocator;
///
/// #[global_allocator]
/// static GLOBAL: TrackingAllocator = TrackingAllocator::new();
/// ```
pub struct TrackingAllocator {
    /// Total number of allocations requested.
    alloc_count: AtomicUsize,
    /// Total number of deallocations requested.
    dealloc_count: AtomicUsize,
    /// Current bytes allocated (approximate, as layout size is provided by caller).
    allocated_bytes: AtomicUsize,
    /// Peak bytes allocated during the lifetime of the program (tracked by this allocator).
    peak_allocated_bytes: AtomicUsize,
    /// Cumulative bytes allocated over the lifetime of the program (monotonic).
    cumulative_allocated_bytes: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            alloc_count: AtomicUsize::new(0),
            dealloc_count: AtomicUsize::new(0),
            allocated_bytes: AtomicUsize::new(0),
            peak_allocated_bytes: AtomicUsize::new(0),
            cumulative_allocated_bytes: AtomicUsize::new(0),
        }
    }

    /// Returns a snapshot of the current memory statistics.
    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            dealloc_count: self.dealloc_count.load(Ordering::Relaxed),
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            peak_allocated_bytes: self.peak_allocated_bytes.load(Ordering::Relaxed),
            cumulative_allocated_bytes: self.cumulative_allocated_bytes.load(Ordering::Relaxed),
        }
    }
    
    /// Resets the counters (use with caution, primarily for scoped measurements).
    /// Note: This does not free memory, just resets the stats counters.
    /// Peak allocation will reset to current allocation.
    pub fn reset_counts(&self) {
        self.alloc_count.store(0, Ordering::Relaxed);
        self.dealloc_count.store(0, Ordering::Relaxed);
        // We do not reset allocated_bytes to 0 because memory is still held.
        // We might want a "diff" instead of a reset in the profiler.
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
            let size = layout.size();
            let prev_bytes = self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
            self.cumulative_allocated_bytes.fetch_add(size, Ordering::Relaxed);
            let current_bytes = prev_bytes + size;
            
            // Update peak if necessary.
            // A loop is required for atomic max.
            let mut peak = self.peak_allocated_bytes.load(Ordering::Relaxed);
            while current_bytes > peak {
                match self.peak_allocated_bytes.compare_exchange_weak(
                    peak,
                    current_bytes,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new_peak) => peak = new_peak,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AllocatorStats {
    pub alloc_count: usize,
    pub dealloc_count: usize,
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub cumulative_allocated_bytes: usize,
}

impl AllocatorStats {
    /// Calculates the difference between two snapshots.
    /// `other` is the "start" snapshot, `self` is the "end" snapshot.
    pub fn diff(&self, start: &AllocatorStats) -> AllocatorStats {
        AllocatorStats {
            alloc_count: self.alloc_count.saturating_sub(start.alloc_count),
            dealloc_count: self.dealloc_count.saturating_sub(start.dealloc_count),
            // For bytes, we might have allocated more or less.
            // If we just want "net change", we subtract.
            // But usually for profiling a block, we want "how much did this block allocate?"
            // The `allocated_bytes` field in the allocator tracks *current* usage.
            // So (End - Start) = Net Growth.
            allocated_bytes: self.allocated_bytes.saturating_sub(start.allocated_bytes), // Kept as usize saturating for internal stat diffs if needed, but ProfileReport uses isize logic.
            // Peak during the interval is harder to track with just start/end snapshots 
            // if the peak happened in the middle and dropped down.
            // The `peak_allocated_bytes` in the allocator is a global high-water mark.
            peak_allocated_bytes: self.peak_allocated_bytes, 
            cumulative_allocated_bytes: self.cumulative_allocated_bytes.saturating_sub(start.cumulative_allocated_bytes),
        }
    }
}
