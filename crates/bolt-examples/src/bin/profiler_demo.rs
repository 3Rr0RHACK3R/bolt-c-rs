use bolt_cpu::CpuBackend;
use bolt_profiler::{ProfiledBackend, TrackingAllocator};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() {
    println!("Starting Profiler Demo...\n");

    let backend: Arc<ProfiledBackend<CpuBackend>> = Arc::new(
        ProfiledBackend::builder(CpuBackend::new())
            .with_tracking_allocator(&GLOBAL)
            .build(),
    );

    // Example 1: Heavy Allocation within scope
    println!("--- Profile: Heavy Allocation ---");
    backend.begin_scope("heavy_allocation");
    {
        let mut vec = Vec::new();
        for i in 0..1_000_000 {
            vec.push(i);
        }
        std::hint::black_box(&vec);
    }
    let report = backend.end_scope().expect("scope");

    println!("Wall Time: {:?}", report.time.host.wall_time);
    println!("Alloc Count: {}", report.memory.host.alloc_count);
    println!(
        "Scope Peak: {:.2} MB",
        report.memory.host.peak_in_scope as f64 / 1024.0 / 1024.0
    );

    // Example 2: CPU Bound
    println!("\n--- Profile: CPU Bound ---");
    backend.begin_scope("cpu_bound");
    {
        let mut sum = 0u64;
        for i in 0..100_000_000 {
            sum = sum.wrapping_add(i);
        }
        std::hint::black_box(sum);
    }
    let report = backend.end_scope().expect("scope");

    println!("Wall Time: {:?}", report.time.host.wall_time);
    println!("Thread CPU: {:?}", report.time.host.thread_time);
    println!("User CPU (process): {:?}", report.time.host.user_time);

    // Example 3: Nested Profiling
    println!("\n--- Profile: Nested Scopes ---");
    backend.begin_scope("outer_scope");
    {
        println!("Inside outer block...");

        backend.begin_scope("inner_scope");
        {
            let _v = vec![0u8; 1024 * 1024 * 10]; // 10MB
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let inner_report = backend.end_scope().expect("inner scope");

        println!(
            "  -> Inner: time={:?}, scope_peak={:.2} MB",
            inner_report.time.host.wall_time,
            inner_report.memory.host.peak_in_scope as f64 / 1024.0 / 1024.0
        );
    }
    let outer_report = backend.end_scope().expect("outer scope");

    println!("Outer Total Time: {:?}", outer_report.time.host.wall_time);
    println!(
        "Outer Scope Peak: {:.2} MB",
        outer_report.memory.host.peak_in_scope as f64 / 1024.0 / 1024.0
    );

    // Show hierarchical report
    println!("\n--- Hierarchical Report ---");
    bolt_profiler::print_report(backend.registry());
}
