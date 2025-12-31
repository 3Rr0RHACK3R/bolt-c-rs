use std::fs;

use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, Error, LoadOpts, OnError, Record, RecordMeta, Role, SaveOpts, load_checkpoint,
    save_checkpoint,
};
use tempfile::TempDir;

fn make_record(name: &str, byte_len: usize) -> Record<'static> {
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
fn checksum_fail_vs_skip() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("checksum");

    save_checkpoint(
        [Ok(make_record("a", 1000)), Ok(make_record("b", 1000))],
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            shard_max_bytes: Some(1000),
            checksum: true,
            ..Default::default()
        },
    )?;

    let shards_dir = out_dir.join("shards");
    let mut shard_files: Vec<_> = fs::read_dir(&shards_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_files.sort();

    assert_eq!(shard_files.len(), 2);

    let corrupt_path = &shard_files[0];
    let mut bytes = fs::read(corrupt_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0b0000_0001;
    fs::write(corrupt_path, bytes).unwrap();

    let strict = load_checkpoint(
        &out_dir,
        &LoadOpts {
            lazy: true,
            on_error: OnError::Fail,
        },
    );
    assert!(matches!(strict, Err(Error::ShardChecksumMismatch { .. })));

    let ckpt = load_checkpoint(
        &out_dir,
        &LoadOpts {
            lazy: true,
            on_error: OnError::Skip,
        },
    )?;

    assert!(matches!(
        ckpt.get("a"),
        Err(Error::RecordUnavailable { .. })
    ));
    assert!(ckpt.get("b").is_ok());

    Ok(())
}
