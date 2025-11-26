use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_profiler::{ProfiledBackend, TrackingAllocator};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Profiler with Decorator Pattern...\n");

    let backend = Arc::new(
        ProfiledBackend::builder(CpuBackend::new())
            .with_tracking_allocator(&GLOBAL)
            .build(),
    );

    let shape = [1000, 1000];
    let size = 1000 * 1000;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();

    // Tensor Creation
    let a = Tensor::from_slice(&backend, &data, &shape)?;
    let b = Tensor::from_slice(&backend, &data, &shape)?;

    // Run Ops
    let _c = a.add(&b)?;

    // Smaller for MatMul speed
    let small_shape = [128, 128];
    let small_data = vec![1.0; 128 * 128];
    let t1 = Tensor::from_slice(&backend, &small_data, &small_shape)?;
    let t2 = Tensor::from_slice(&backend, &small_data, &small_shape)?;
    let _t3 = t1.matmul(&t2)?;

    println!("=== Default Summary (top-level only) ===");
    backend.print_report();

    println!("\n=== Detailed Summary (with hierarchy) ===");
    backend.print_report_detailed();

    println!("\n=== Memory Details ===");
    backend.print_memory_details();

    // Demonstrate scoped profiling
    println!("\n=== Scoped Profiling Demo ===");
    backend.clear_stats();

    backend.begin_scope("forward_pass");
    let x = Tensor::from_slice(&backend, &data, &shape)?;
    let y = Tensor::from_slice(&backend, &data, &shape)?;
    let _z = x.add(&y)?;
    let report = backend.end_scope().expect("scope report");

    println!("Scope 'forward_pass' completed:");
    println!("  Wall time: {:?}", report.wall_time);
    println!("  Allocs: {}", report.memory_stats.alloc_count);
    println!("  Deallocs: {}", report.memory_stats.dealloc_count);
    println!("  Peak in scope: {} bytes", report.memory_stats.peak_in_scope);

    backend.print_report_detailed();

    Ok(())
}
