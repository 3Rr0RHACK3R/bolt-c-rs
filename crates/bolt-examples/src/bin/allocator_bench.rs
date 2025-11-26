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
    
    println!("Total Time: {:.2?}", duration);
    println!("Average: {:.2?} per alloc/free cycle", duration / iterations);
    println!("Throughput: {:.2} allocs/sec", iterations as f64 / duration.as_secs_f64());
    println!("System Allocations: {}", stats.alloc_count);
    println!("System Deallocations: {}", stats.dealloc_count);
    println!();
}

fn main() {
    println!("Allocator Benchmark: Naive vs Caching (Pool)");
    println!("Iterations: 2,000 | Allocation Size: 10MB");
    println!("=================================================\n");
    
    // 1. Naive (System Allocator)
    let naive = CpuBackend::new();
    benchmark_allocator(naive, "Naive Allocator (System Malloc)");

    // 2. Caching (Pool)
    let caching = CpuBackend::with_pooling();
    benchmark_allocator(caching, "Caching Allocator (Memory Pool)");
}