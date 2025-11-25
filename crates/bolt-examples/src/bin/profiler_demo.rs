use bolt_profiler::{profile, TrackingAllocator};

// 1. Register the Tracking Allocator
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() {
    println!("Starting Profiler Demo...");

    // Example 1: Heavy Allocation
    println!("\n--- Profile: Heavy Allocation ---");
    let (_, report) = profile(Some(&GLOBAL), || {
        let mut vec = Vec::new();
        for i in 0..1_000_000 {
            vec.push(i);
        }
        // Keep it alive to ensure peak memory reflects it
        std::hint::black_box(&vec);
    });

    println!("Wall Time: {:?}", report.wall_time);
    println!("Alloc Count: {}", report.memory_stats.alloc_count);
    println!("Total Allocated: {} MB (Cumulative)", report.memory_stats.total_allocated_bytes as f64 / 1024.0 / 1024.0);
    println!("Net Allocated: {} MB", report.memory_stats.net_allocated_bytes as f64 / 1024.0 / 1024.0);
    println!("Peak RSS: {} MB", report.memory_stats.peak_rss_bytes as f64 / 1024.0 / 1024.0);
    println!("Peak Allocated (Heap): {} MB", report.memory_stats.peak_allocated_bytes as f64 / 1024.0 / 1024.0);

    // Example 2: CPU Bound
    println!("\n--- Profile: CPU Bound ---");
    let (_, report) = profile(Some(&GLOBAL), || {
        let mut sum = 0u64;
        for i in 0..100_000_000 {
            sum = sum.wrapping_add(i);
        }
        std::hint::black_box(sum);
    });

    println!("Wall Time: {:?}", report.wall_time);
    println!("User CPU: {:?}", report.cpu_stats.user_time);
    println!("Alloc Count: {} (Should be close to 0)", report.memory_stats.alloc_count);

    // Example 3: Nested Profiling (Simulating Op)
    println!("\n--- Profile: Nested / Op Simulation ---");
    let (_, outer_report) = profile(Some(&GLOBAL), || {
        println!("Inside outer block...");
        
        let (_, inner_report) = profile(Some(&GLOBAL), || {
            let _v = vec![0u8; 1024 * 1024 * 10]; // 10MB
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
        
        println!("  -> Inner Op: Time={:?}, Net Alloc={} MB, Total Alloc={} MB", 
            inner_report.wall_time, 
            inner_report.memory_stats.net_allocated_bytes as f64 / 1024.0 / 1024.0,
            inner_report.memory_stats.total_allocated_bytes as f64 / 1024.0 / 1024.0
        );
    });

    println!("Outer Total Time: {:?}", outer_report.wall_time);
    println!("Outer Net Alloc: {} MB", outer_report.memory_stats.net_allocated_bytes as f64 / 1024.0 / 1024.0);
}
