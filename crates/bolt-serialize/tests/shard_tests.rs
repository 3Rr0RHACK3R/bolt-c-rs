use std::fs;

use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, LoadOpts, Record, RecordMeta, Role, SaveOpts, load_checkpoint, save_checkpoint,
};
use tempfile::TempDir;

fn make_record(name: &str, byte_len: usize) -> Record<'static> {
    debug_assert!(
        byte_len % 4 == 0,
        "byte_len must be divisible by 4 for F32 dtype"
    );
    Record::new(
        RecordMeta::new(
            name,
            DType::F32,
            Shape::from_slice(&[byte_len / 4]).unwrap(),
        )
        .with_role(Role::User),
        vec![0u8; byte_len],
    )
}

#[test]
fn shard_integration_via_checkpoint_save() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test_sharding");

    let records = vec![
        make_record("c_tensor", 100),
        make_record("a_tensor", 100),
        make_record("b_tensor", 100),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let a_view = ckpt.get("a_tensor")?;
    assert_eq!(a_view.shape.as_slice(), &[25]);
    assert_eq!(a_view.data.len(), 100);

    let b_view = ckpt.get("b_tensor")?;
    assert_eq!(b_view.shape.as_slice(), &[25]);
    assert_eq!(b_view.data.len(), 100);

    let c_view = ckpt.get("c_tensor")?;
    assert_eq!(c_view.shape.as_slice(), &[25]);
    assert_eq!(c_view.data.len(), 100);

    Ok(())
}

#[test]
fn many_records_creates_shards() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test_many_shards");

    let records: Vec<_> = (0..10)
        .map(|i| make_record(&format!("tensor_{:03}", i), 1000))
        .collect();

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            shard_max_bytes: Some(2000),
            ..Default::default()
        },
    )?;

    let shards_dir = out_dir.join("shards");
    assert!(shards_dir.exists());

    let shard_files: Vec<_> = fs::read_dir(&shards_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "safetensors")
        })
        .collect();

    assert!(
        shard_files.len() > 1,
        "expected multiple shards, got {}",
        shard_files.len()
    );

    Ok(())
}
