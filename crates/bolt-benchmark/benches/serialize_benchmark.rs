use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, Record, RecordMeta, Role, SaveOpts,
    save_checkpoint,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

fn make_record(name: &str, numel: usize) -> Record<'static> {
    let meta = RecordMeta::new(name, DType::F32, Shape::from_slice(&[numel]).unwrap())
        .with_role(Role::Param);
    let nbytes = meta.nbytes().unwrap() as usize;
    Record::new(meta, vec![0u8; nbytes])
}

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
                
                let records: Vec<_> = (0..count)
                    .map(|i| make_record(&format!("tensor_{i:03}"), tensor_size))
                    .collect();
                
                save_checkpoint(
                    records.into_iter().map(Ok),
                    &out_dir,
                    &CheckpointMeta::default(),
                    &SaveOpts::default(),
                ).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_save_checkpoint);
criterion_main!(benches);
