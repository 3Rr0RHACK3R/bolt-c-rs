use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, Record, RecordMeta, Result, SaveOpts,
    save_checkpoint,
};
use tempfile::TempDir;

fn make_large_record(name: &str, size_mb: usize) -> Record<'static> {
    let numel = size_mb * 1024 * 1024 / 4;
    Record::new(
        RecordMeta::new(name, DType::F32, Shape::from_slice(&[numel]).unwrap()),
        vec![0u8; numel * 4],
    )
}

#[test]
fn large_checkpoint_write() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("large_checkpoint");
    
    let records = vec![
        make_large_record("tensor_0", 10),
        make_large_record("tensor_1", 10),
    ];
    
    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;
    
    assert!(out_dir.join("bolt-checkpoint.json").exists());
    assert!(out_dir.join("shards").exists());
    
    Ok(())
}
