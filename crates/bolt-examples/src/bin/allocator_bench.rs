use bolt_core::allocator::StorageAllocator;
use bolt_core::backend::Backend;
use bolt_core::dtype::DType;
use bolt_cpu::backend::CpuBackend;
use bolt_profiler::TrackingAllocator;
use std::time::Instant;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn benchmark_allocator(backend: CpuBackend, name: &str) {
    let allocator = <CpuBackend as Backend<f32>>::allocator(&backend);
    
    // Reset global allocator stats before run
    GLOBAL.reset_counts();
    let baseline = GLOBAL.stats();
    
    let start = Instant::now();
    let iterations = 2_000;
    // 10MB - larger tensor size where mmap cost is significant
    let size_bytes = 10 * 1024 * 1024; 

    println!("--- {} ---", name);
    for _ in 0..iterations {
        let mut t = allocator.allocate_bytes(size_bytes, DType::F32).unwrap();
        // Force a write to ensure memory is actually touched (OS page fault simulation)
        // In caching allocator, memory is already paged in after first use.
        unsafe {
            let slice = t.try_as_mut_slice().unwrap();
            slice[0] = 1.0;
        }
    }
    
    let duration = start.elapsed();
    let stats = GLOBAL.stats();
    let alloc_delta = stats.alloc_count;
    let dealloc_delta = stats.dealloc_count;
    let bytes_delta = stats
        .cumulative_allocated_bytes
        .saturating_sub(baseline.cumulative_allocated_bytes);
    let live_delta = stats
        .allocated_bytes
        .saturating_sub(baseline.allocated_bytes);
    
    println!("Total Time: {:.2?}", duration);
    println!("Average: {:.2?} per alloc/free cycle", duration / iterations);
    println!("Throughput: {:.2} allocs/sec", iterations as f64 / duration.as_secs_f64());
    println!("System Allocations: {}", alloc_delta);
    println!("System Deallocations: {}", dealloc_delta);
    println!("Bytes Allocated (delta): {}", bytes_delta);
    println!("Live Bytes Delta: {}", live_delta);
    println!();
}

fn main() {
    println!("Allocator Benchmark: Naive vs Caching (Pool)");
    println!("Iterations: 2,000 | Allocation Size: 10MB");
    println!("=================================================\n");
    
    // 1. Naive (System Allocator) - single run
    let naive = CpuBackend::new();
    benchmark_allocator(naive, "Naive Allocator (System Malloc)");
 
    // 2. Caching (Pool) - multiple rounds on the same backend instance
    let caching = CpuBackend::with_pooling();
    let rounds = 5;
    for round in 1..=rounds {
        let label = format!("Caching Allocator (Memory Pool) [round {}]", round);
        benchmark_allocator(caching.clone(), &label);
    }
}
