use bolt_cpu::CpuBackend;
use bolt_profiler::{HostMemTracker, ProfiledBackend};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: HostMemTracker = HostMemTracker::new();

fn main() {
    println!("Starting Profiler Demo...\n");

    let (backend, profiler) = ProfiledBackend::wrap_with_host_mem(CpuBackend::new(), &GLOBAL);
    let backend: Arc<ProfiledBackend<CpuBackend>> = Arc::new(backend);

    // Example 1: Heavy Allocation within scope
    println!("--- Profile: Heavy Allocation ---");
    profiler.with_scope("heavy_allocation", || {
        // Large tensor creation to exercise host/device allocs.
        let shape = [2048, 2048];
        let len = shape.iter().product();
        let data: Vec<f32> = (0..len).map(|i| (i % 1024) as f32).collect();
        let t = bolt_tensor::Tensor::from_slice(&backend, &data, &shape).unwrap();
        let _ = t.add(&t).unwrap();
    });

    // Example 2: CPU Bound
    println!("\n--- Profile: CPU Bound ---");
    profiler.with_scope("cpu_bound", || {
        let shape = [1024, 1024];
        let len = shape.iter().product();
        let data: Vec<f32> = (0..len).map(|i| (i % 512) as f32 * 0.001).collect();
        let a = bolt_tensor::Tensor::from_slice(&backend, &data, &shape).unwrap();
        let b = bolt_tensor::Tensor::from_slice(&backend, &data, &shape).unwrap();
        let _ = a.matmul(&b).unwrap();
    });

    // Example 3: Nested Profiling
    println!("\n--- Profile: Nested Scopes ---");
    profiler.with_scope("outer_scope", || {
        println!("Inside outer block...");

        profiler.with_scope("inner_scope", || {
            let shape = [512, 512];
            let len = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| (i % 256) as f32).collect();
            let t1 = bolt_tensor::Tensor::from_slice(&backend, &data, &shape).unwrap();
            let t2 = bolt_tensor::Tensor::from_slice(&backend, &data, &shape).unwrap();
            let _ = t1.add(&t2).unwrap();
        });
    });

    // Show hierarchical report
    println!("\n--- Hierarchical Report ---");
    bolt_profiler::print_report(profiler.registry());
}
