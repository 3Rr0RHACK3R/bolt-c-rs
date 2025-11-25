use bolt_profiler::{profile, TrackingAllocator};

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
        std::hint::black_box(&vec);
    });

    println!("Wall Time: {:?}", report.wall_time);
    println!("Thread CPU: {:?}", report.cpu_stats.thread_time);
    println!("Alloc Count: {}", report.memory_stats.alloc_count);
    println!(
        "Total Allocated: {:.2} MB",
        report.memory_stats.total_allocated_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "Scope Peak: {:.2} MB",
        report.memory_stats.scope_peak_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "Net Allocated: {:.2} MB",
        report.memory_stats.net_allocated_bytes as f64 / 1024.0 / 1024.0
    );

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
    println!("Thread CPU: {:?}", report.cpu_stats.thread_time);
    println!("User CPU (process): {:?}", report.cpu_stats.user_time);
    println!(
        "Alloc Count: {} (should be ~0)",
        report.memory_stats.alloc_count
    );

    // Example 3: Nested Profiling
    println!("\n--- Profile: Nested / Op Simulation ---");
    let (_, outer_report) = profile(Some(&GLOBAL), || {
        println!("Inside outer block...");

        let (_, inner_report) = profile(Some(&GLOBAL), || {
            let _v = vec![0u8; 1024 * 1024 * 10]; // 10MB
            std::thread::sleep(std::time::Duration::from_millis(50));
        });

        println!(
            "  -> Inner: time={:?}, scope_peak={:.2} MB",
            inner_report.wall_time,
            inner_report.memory_stats.scope_peak_bytes as f64 / 1024.0 / 1024.0
        );
    });

    println!("Outer Total Time: {:?}", outer_report.wall_time);
    println!(
        "Outer Scope Peak: {:.2} MB",
        outer_report.memory_stats.scope_peak_bytes as f64 / 1024.0 / 1024.0
    );
}
