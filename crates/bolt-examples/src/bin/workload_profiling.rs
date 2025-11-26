use std::sync::Arc;

use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_profiler::{ProfiledBackend, TrackingAllocator};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() -> AnyResult<()> {
    let backend = Arc::new(
        ProfiledBackend::builder(CpuBackend::new())
            .with_tracking_allocator(&GLOBAL)
            .build(),
    );

    run_scoped_workload("dense_matmul", &backend, run_dense_matmul)?;
    run_scoped_workload("add_chain", &backend, run_add_chain)?;
    run_scoped_workload("mean_reduction", &backend, run_mean_reduction)?;
    run_scoped_workload("transpose_copy", &backend, run_transpose_copy)?;

    println!("\n===== All workloads summary =====");
    backend.print_report_detailed();

    Ok(())
}

fn run_scoped_workload<F>(
    label: &str,
    backend: &Arc<ProfiledBackend<CpuBackend>>,
    workload: F,
) -> AnyResult<()>
where
    F: FnOnce(&Arc<ProfiledBackend<CpuBackend>>) -> AnyResult<()>,
{
    backend.begin_scope(label);
    workload(backend)?;
    let report = backend.end_scope().expect("scope report");

    println!(
        "[{}] time={:?}, allocs={}, peak={}",
        label,
        report.wall_time,
        report.memory_stats.alloc_count,
        report.memory_stats.peak_in_scope
    );
    Ok(())
}

fn run_dense_matmul(backend: &Arc<ProfiledBackend<CpuBackend>>) -> AnyResult<()> {
    let shape = [1024, 1024];
    let len = shape.iter().product();
    let data = make_data(len, 0.001);
    let a = Tensor::from_slice(backend, &data, &shape)?;
    let b = Tensor::from_slice(backend, &data, &shape)?;
    let _ = a.matmul(&b)?;
    Ok(())
}

fn run_add_chain(backend: &Arc<ProfiledBackend<CpuBackend>>) -> AnyResult<()> {
    let shape = [4096, 256];
    let len = shape.iter().product();
    let data = make_data(len, 0.0003);
    let base = Tensor::from_slice(backend, &data, &shape)?;
    let mut acc = base.clone();
    for _ in 0..6 {
        acc = acc.add(&base)?;
    }
    let _ = acc.sub(&base)?;
    Ok(())
}

fn run_mean_reduction(backend: &Arc<ProfiledBackend<CpuBackend>>) -> AnyResult<()> {
    let shape = [1024, 1024];
    let len = shape.iter().product();
    let data = make_data(len, 0.0001);
    let tensor = Tensor::from_slice(backend, &data, &shape)?;
    let _ = tensor.mean_f32()?;
    Ok(())
}

fn run_transpose_copy(backend: &Arc<ProfiledBackend<CpuBackend>>) -> AnyResult<()> {
    let shape = [2048, 256];
    let len = shape.iter().product();
    let data = make_data(len, 0.0002);
    let tensor = Tensor::from_slice(backend, &data, &shape)?;
    let view = tensor.transpose(0, 1)?;
    let _contig = view.contiguous()?;
    Ok(())
}

fn make_data(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 1024) as f32 - 512.0) * scale)
        .collect()
}
