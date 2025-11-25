use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_profiler::{ProfiledBackend, TrackingAllocator};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Profiler with Decorator Pattern...\n");

    // 1. Wrap the backend
    // We use ProfiledBackend<CpuBackend> to intercept calls.
    let backend = Arc::new(ProfiledBackend::new(CpuBackend::new(), Some(&GLOBAL)));

    // Write normal Tensor code
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

    backend.print_report();

    Ok(())
}

