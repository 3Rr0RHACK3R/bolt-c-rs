use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize::{CheckpointMeta, CheckpointOptions, save};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

type B = CpuBackend;
type D = f32;

fn bench_save_checkpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_save");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    let tensor_counts = [10, 50, 100];
    let tensor_size = 1024 * 256;

    for &count in &tensor_counts {
        let id = BenchmarkId::new("records", format!("{count}_tensors"));

        group.bench_function(id, |b| {
            let tmp = TempDir::new().unwrap();
            let mut counter = 0u64;

            b.iter(|| {
                let out_dir = tmp.path().join(format!("ckpt_{counter}"));
                counter += 1;

                let backend = Arc::new(CpuBackend::new());
                let store = Store::<B, D>::new(backend.clone(), counter);

                for i in 0..count {
                    store
                        .param(&format!("tensor_{i:03}"), &[tensor_size], Init::Zeros)
                        .unwrap();
                }
                store.seal();

                save(
                    &store,
                    &out_dir,
                    &CheckpointMeta::default(),
                    &CheckpointOptions::default(),
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_save_checkpoint);
criterion_main!(benches);
