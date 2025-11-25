use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct TrackingAllocator {
    alloc_count: AtomicUsize,
    dealloc_count: AtomicUsize,
    allocated_bytes: AtomicUsize,
    peak_allocated_bytes: AtomicUsize,
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

    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            dealloc_count: self.dealloc_count.load(Ordering::Relaxed),
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            peak_allocated_bytes: self.peak_allocated_bytes.load(Ordering::Relaxed),
            cumulative_allocated_bytes: self.cumulative_allocated_bytes.load(Ordering::Relaxed),
        }
    }

    pub fn reset_counts(&self) {
        self.alloc_count.store(0, Ordering::Relaxed);
        self.dealloc_count.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
            let size = layout.size();
            let prev_bytes = self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
            self.cumulative_allocated_bytes
                .fetch_add(size, Ordering::Relaxed);
            let current_bytes = prev_bytes + size;

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
        self.allocated_bytes
            .fetch_sub(layout.size(), Ordering::Relaxed);
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
