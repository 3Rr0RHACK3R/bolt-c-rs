use bolt_profiler::{profile, TrackingAllocator};
use std::alloc::GlobalAlloc;
use std::alloc::Layout;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

#[test]
fn test_allocator_accuracy() {
    // This test verifies that the TrackingAllocator correctly counts bytes and calls.
    // Note: We must run this in a separate process or ensure no other threads are allocating
    // to be strictly precise, but for unit tests `cargo test` runs in parallel.
    // However, since we are checking *deltas* inside a profile block, and `profile` 
    // snapshots global state, we might get noise from other threads.
    // Ideally, we use `--test-threads=1` for this, but let's try to be robust.
    
    // We allocate a substantial amount to distinguish from background noise.
    let alloc_size = 1024 * 1024; // 1MB

    let (_, report) = profile(Some(&GLOBAL), || {
        // Use GlobalAlloc directly to bypass Vec resizing logic / capacity over-allocation
        let layout = Layout::from_size_align(alloc_size, 1).unwrap();
        unsafe {
            let ptr = GLOBAL.alloc(layout);
            assert!(!ptr.is_null());
            // Prevent optimization
            std::ptr::write_volatile(ptr, 0xFF);
            GLOBAL.dealloc(ptr, layout);
        }
    });

    // Verify Alloc Count
    // We expect exactly 1 alloc and 1 dealloc if we called them manually.
    assert_eq!(report.memory_stats.alloc_count, 1, "Should record exactly 1 allocation");
    assert_eq!(report.memory_stats.dealloc_count, 1, "Should record exactly 1 deallocation");

    // Verify Bytes
    assert_eq!(report.memory_stats.total_allocated_bytes, alloc_size, "Total allocated should match request");
    assert_eq!(report.memory_stats.net_allocated_bytes, 0, "Net allocation should be 0 after dealloc");
}

#[test]
fn test_vec_behavior() {
    // Vec allocation might be larger than requested due to capacity strategies.
    let (_, report) = profile(Some(&GLOBAL), || {
        let mut v: Vec<u8> = Vec::with_capacity(1000);
        v.push(1);
    });

    assert!(report.memory_stats.alloc_count >= 1);
    assert!(report.memory_stats.total_allocated_bytes >= 1000);
    // Net bytes > 0 because `v` is dropped *after* the profile closure returns?
    // Wait, `profile` runs the closure `f()`. `v` is dropped at the end of the closure scope.
    // So `v` should be deallocated *before* the end snapshot?
    // YES. `v` goes out of scope at the closing brace of the closure.
    // So net allocated should be 0.
    
    assert_eq!(report.memory_stats.net_allocated_bytes, 0, "Vec should be dropped inside closure");
}

#[test]
fn test_leak_behavior() {
    let (_, report) = profile(Some(&GLOBAL), || {
        let v: Vec<u8> = Vec::with_capacity(500);
        std::mem::forget(v); // Leak it
    });

    assert_eq!(report.memory_stats.net_allocated_bytes, 500);
}
