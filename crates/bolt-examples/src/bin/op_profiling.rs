use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_profiler::{HostMemTracker, ProfiledBackend};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: HostMemTracker = HostMemTracker::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Profiler with Decorator Pattern...\n");

    let (backend, profiler) = ProfiledBackend::wrap_with_host_mem(CpuBackend::new(), &GLOBAL);
    let backend = Arc::new(backend);
    let run_forward =
        |backend: &Arc<ProfiledBackend<CpuBackend>>| -> Result<(), Box<dyn std::error::Error>> {
            let shape = [1000, 1000];
            let size = 1000 * 1000;
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();

            let a = Tensor::from_slice(backend, &data, &shape)?;
            let b = Tensor::from_slice(backend, &data, &shape)?;
            let _c = a.add(&b)?;
            Ok(())
        };

    let small_shape = [128, 128];
    let small_data = vec![1.0; 128 * 128];
    let t1 = Tensor::from_slice(&backend, &small_data, &small_shape)?;
    let t2 = Tensor::from_slice(&backend, &small_data, &small_shape)?;
    let _t3 = t1.matmul(&t2)?;

    println!("=== Default Summary (top-level only) ===");
    bolt_profiler::print_report(profiler.registry());

    // Scoped profiling
    println!("\n=== Scoped Profiling Demo ===");
    profiler.clear();
    let scoped = profiler.with_scope("forward_pass", || run_forward(&backend));
    scoped?;

    bolt_profiler::print_report(profiler.registry());

    Ok(())
}
