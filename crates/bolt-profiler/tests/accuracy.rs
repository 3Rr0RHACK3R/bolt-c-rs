use bolt_profiler::{TrackingAllocator, profile};
use serial_test::serial;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

#[test]
#[serial]
fn test_allocator_accuracy() {
    // This test verifies that the TrackingAllocator correctly counts bytes and calls.
    // Note: We must run this in a separate process or ensure no other threads are allocating
    // to be strictly precise, but for unit tests `cargo test` runs in parallel.
    // However, since we are checking *deltas* inside a profile block, and `profile`
    // snapshots global state, we might get noise from other threads.
    // Ideally, we use `--test-threads=1` for this, but let's try to be robust.

    // We allocate a substantial amount to distinguish from background noise.
    let alloc_size = 1024 * 1024; // 1MB
    let alloc = TrackingAllocator::new();

    let (_, report) = profile(Some(&alloc), || {
        // Use TrackingAllocator directly to bypass Vec resizing logic / capacity over-allocation
        let layout = Layout::from_size_align(alloc_size, 1).unwrap();
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            // Prevent optimization
            std::ptr::write_volatile(ptr, 0xFF);
            alloc.dealloc(ptr, layout);
        }
    });

    // Verify Alloc Count
    // We expect exactly 1 alloc and 1 dealloc if we called them manually.
    assert_eq!(
        report.memory_stats.alloc_count, 1,
        "Should record exactly 1 allocation"
    );
    assert_eq!(
        report.memory_stats.dealloc_count, 1,
        "Should record exactly 1 deallocation"
    );

    // Verify Bytes
    assert_eq!(
        report.memory_stats.total_allocated_bytes, alloc_size,
        "Total allocated should match request"
    );
    assert_eq!(
        report.memory_stats.net_allocated_bytes, 0,
        "Net allocation should be 0 after dealloc"
    );
}

#[test]
#[serial]
fn test_vec_behavior() {
    // Simulate growth-style allocations and ensure drops happen within the profile scope.
    let alloc = TrackingAllocator::new();
    let layout_a = Layout::from_size_align(1_000, 1).unwrap();
    let layout_b = Layout::from_size_align(2_000, 1).unwrap();

    let (_, report) = profile(Some(&alloc), || unsafe {
        let ptr_a = alloc.alloc(layout_a);
        assert!(!ptr_a.is_null());
        let ptr_b = alloc.alloc(layout_b);
        assert!(!ptr_b.is_null());
        alloc.dealloc(ptr_b, layout_b);
        alloc.dealloc(ptr_a, layout_a);
    });

    assert_eq!(report.memory_stats.alloc_count, 2);
    assert_eq!(report.memory_stats.dealloc_count, 2);
    assert_eq!(
        report.memory_stats.total_allocated_bytes,
        layout_a.size() + layout_b.size()
    );
    assert_eq!(report.memory_stats.net_allocated_bytes, 0);
}

#[test]
#[serial]
fn test_leak_behavior() {
    let alloc = TrackingAllocator::new();
    let alloc_size = 500;
    let layout = Layout::from_size_align(alloc_size, 1).unwrap();
    let (_, report) = profile(Some(&alloc), || {
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            std::ptr::write_volatile(ptr, 0xAA);
            // Intentionally leak: no dealloc
        }
    });

    assert_eq!(report.memory_stats.net_allocated_bytes, alloc_size as isize);
}

#[test]
#[serial]
fn test_scope_peak_tracking() {
    let alloc = TrackingAllocator::new();
    let (_, report) = profile(Some(&alloc), || {
        // Allocate 1MB, then free, then allocate 512KB
        let layout_1mb = Layout::from_size_align(1024 * 1024, 1).unwrap();
        let layout_512kb = Layout::from_size_align(512 * 1024, 1).unwrap();

        unsafe {
            let ptr1 = alloc.alloc(layout_1mb);
            assert!(!ptr1.is_null());
            std::ptr::write_volatile(ptr1, 0xFF);

            alloc.dealloc(ptr1, layout_1mb);

            let ptr2 = alloc.alloc(layout_512kb);
            assert!(!ptr2.is_null());
            std::ptr::write_volatile(ptr2, 0xAA);

            alloc.dealloc(ptr2, layout_512kb);
        }
    });

    // Peak during scope should be ~1MB (the first allocation)
    assert!(
        report.memory_stats.scope_peak_bytes >= 1024 * 1024,
        "Scope peak should capture the 1MB allocation: got {}",
        report.memory_stats.scope_peak_bytes
    );
    assert_eq!(report.memory_stats.net_allocated_bytes, 0);
}
