use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;

fn bench_gemm_f32(c: &mut Criterion) {
    let backend = Arc::new(CpuBackend::new());
    // Squares + rectangular shapes
    let sizes: &[(usize, usize, usize)] = &[
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (512, 64, 1024),
        (1024, 64, 256),
    ];
    let mut group = c.benchmark_group("gemm_f32");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(3));
    for &(m, k, n) in sizes {
        let id = BenchmarkId::from_parameter(format!("{m}x{k}x{n}"));
        // Deterministic inputs
        let a = Tensor::<CpuBackend, f32>::uniform(&backend, &[m, k], -1.0, 1.0, Some(42)).unwrap();
        let b = Tensor::<CpuBackend, f32>::uniform(&backend, &[k, n], -1.0, 1.0, Some(43)).unwrap();
        let a = a.contiguous().unwrap();
        let b = b.contiguous().unwrap();
        group.bench_with_input(id, &(m, k, n), |bencher, _dims| {
            bencher.iter(|| {
                let out = black_box(a.matmul(&b).unwrap());
                black_box(out.shape()); // touch output
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_f32);
criterion_main!(benches);
