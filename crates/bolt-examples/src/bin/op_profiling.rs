use std::sync::Arc;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_profiler::{profile, TrackingAllocator};

// Register global allocator to track memory
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("| Operation        | Time (µs) | Total (B)   | Net (B)     | Peak RSS (MB) |");
    println!("|------------------|-----------|-------------|-------------|---------------|");

    // 1. Setup Context
    let backend = Arc::new(CpuBackend::new());
    
    // 2. Profile: Tensor Creation (1000x1000 f32 = 4MB)
    let shape = [1000, 1000];
    let size = 1000 * 1000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size).map(|i| i as f32 * 0.002).collect();

    let (tensor_a, report_a) = profile(Some(&GLOBAL), || {
        Tensor::<CpuBackend, f32>::from_slice(&backend, &data_a, &shape)
    });
    print_row("Create Tensor A", &report_a);

    let (tensor_b, report_b) = profile(Some(&GLOBAL), || {
        Tensor::<CpuBackend, f32>::from_slice(&backend, &data_b, &shape)
    });
    print_row("Create Tensor B", &report_b);

    // 3. Profile: Add (A + B) -> C
    // This allocates a new result tensor C (4MB).
    let (tensor_c, report_add) = profile(Some(&GLOBAL), || {
        tensor_a.as_ref().unwrap().add(tensor_b.as_ref().unwrap())
    });
    print_row("Add (A + B)", &report_add);

    // 4. Profile: MatMul (Small)
    // 1000x1000 is too slow for a quick test if unoptimized, let's do smaller
    let small_shape = [128, 128];
    let small_data: Vec<f32> = (0..128*128).map(|_| 1.0).collect();
    let t1 = Tensor::<CpuBackend, f32>::from_slice(&backend, &small_data, &small_shape).unwrap();
    let t2 = Tensor::<CpuBackend, f32>::from_slice(&backend, &small_data, &small_shape).unwrap();

    let (_, report_matmul) = profile(Some(&GLOBAL), || {
        t1.matmul(&t2)
    });
    print_row("MatMul (128x128)", &report_matmul);

    // 5. Profile: Deallocation (Scope Drop)
    let (_, report_drop) = profile(Some(&GLOBAL), || {
        drop(tensor_c);
        // tensor_a and tensor_b are still alive
    });
    // Dealloc count should be > 0
    print_row("Drop Result C", &report_drop);

    Ok(())
}

fn print_row(name: &str, report: &bolt_profiler::ProfileReport) {
    println!(
        "| {:<16} | {:<9} | {:<11} | {:<11} | {:<13.2} |",
        name,
        report.wall_time.as_micros(),
        report.memory_stats.total_allocated_bytes,
        report.memory_stats.net_allocated_bytes,
        report.memory_stats.peak_rss_bytes as f64 / 1024.0 / 1024.0
    );
}
