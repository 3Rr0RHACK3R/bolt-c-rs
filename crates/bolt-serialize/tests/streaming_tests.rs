use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, LoadOpts, Record, RecordMeta, Result, Role, SaveOpts,
    load_checkpoint, save_checkpoint,
};
use tempfile::TempDir;

fn make_record(name: &str, role: Role, byte_len: usize) -> Record<'static> {
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
        .with_role(role),
        vec![0u8; byte_len],
    )
}

fn make_record_with_pattern(name: &str, role: Role, byte_len: usize, pattern: u8) -> Record<'static> {
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
        .with_role(role),
        vec![pattern; byte_len],
    )
}

#[test]
fn checkpoint_roundtrip_with_records() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("roundtrip");

    let records = vec![
        make_record("encoder.weight", Role::Param, 100),
        make_record("decoder.weight", Role::Param, 100),
        make_record("encoder.bias", Role::Param, 100),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let mut names: Vec<_> = ckpt.list().into_iter().map(|m| m.name).collect();
    names.sort();

    assert_eq!(names.len(), 3);
    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(names.contains(&"decoder.weight".to_string()));
    assert!(names.contains(&"encoder.bias".to_string()));

    let view = ckpt.get("encoder.weight")?;
    assert_eq!(view.shape.as_slice(), &[25]);
    assert_eq!(view.data.len(), 100);

    Ok(())
}

#[test]
fn checkpoint_with_exclude_patterns() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude");

    let records = vec![
        make_record("encoder.weight", Role::Param, 100),
        make_record("decoder.weight", Role::Param, 100),
        make_record("encoder.bias", Role::Param, 100),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            exclude: vec!["decoder.*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let names: Vec<_> = ckpt.list().into_iter().map(|m| m.name).collect();

    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(names.contains(&"encoder.bias".to_string()));
    assert!(!names.contains(&"decoder.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_with_checksums() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("checksums");

    let records = vec![
        make_record("tensor_a", Role::Param, 100),
        make_record("tensor_b", Role::Buffer, 200),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            checksum: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    
    let view_a = ckpt.get("tensor_a")?;
    assert_eq!(view_a.data.len(), 100);
    
    let view_b = ckpt.get("tensor_b")?;
    assert_eq!(view_b.data.len(), 200);

    Ok(())
}

#[test]
fn checkpoint_multiple_shards() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("shards");

    let records: Vec<_> = (0..10)
        .map(|i| make_record(&format!("tensor_{:03}", i), Role::Param, 1000))
        .collect();

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            shard_max_bytes: Some(2000),
            ..Default::default()
        },
    )?;

    let shards_dir = out_dir.join("shards");
    assert!(shards_dir.exists());

    let shard_files: Vec<_> = std::fs::read_dir(&shards_dir)
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

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    for i in 0..10 {
        let name = format!("tensor_{:03}", i);
        let view = ckpt.get(&name)?;
        assert_eq!(view.data.len(), 1000);
    }

    Ok(())
}

#[test]
fn checkpoint_with_pattern_data() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("pattern_data");

    let records = vec![
        make_record_with_pattern("data_a", Role::Param, 400, 0xAA),
        make_record_with_pattern("data_b", Role::Buffer, 800, 0xBB),
        make_record_with_pattern("data_c", Role::User, 200, 0xCC),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    
    let view_a = ckpt.get("data_a")?;
    assert_eq!(view_a.data.len(), 400);
    assert!(view_a.data.iter().all(|&b| b == 0xAA));
    
    let view_b = ckpt.get("data_b")?;
    assert_eq!(view_b.data.len(), 800);
    assert!(view_b.data.iter().all(|&b| b == 0xBB));
    
    let view_c = ckpt.get("data_c")?;
    assert_eq!(view_c.data.len(), 200);
    assert!(view_c.data.iter().all(|&b| b == 0xCC));

    Ok(())
}

#[test]
fn checkpoint_preserves_metadata() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("metadata");

    let user_data = serde_json::json!({
        "commit": "abc123",
        "lr": 0.001
    });

    let meta = CheckpointMeta {
        epoch: Some(42),
        global_step: Some(123456),
        model_name: Some("TestModel".to_string()),
        user: user_data.clone(),
    };

    save_checkpoint(
        std::iter::once(Ok(make_record("test", Role::Param, 100))),
        &out_dir,
        &meta,
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    assert_eq!(ckpt.info().meta.epoch, Some(42));
    assert_eq!(ckpt.info().meta.global_step, Some(123456));
    assert_eq!(ckpt.info().meta.model_name, Some("TestModel".to_string()));
    assert_eq!(ckpt.info().meta.user["commit"], "abc123");
    assert_eq!(ckpt.info().meta.user["lr"], 0.001);

    Ok(())
}

#[test]
fn checkpoint_roles_preserved() -> Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("roles");

    save_checkpoint(
        [
            make_record("param", Role::Param, 100),
            make_record("buffer", Role::Buffer, 100),
            make_record("optim", Role::Optimizer, 100),
            make_record("rng", Role::Rng, 100),
            make_record("user", Role::User, 100),
        ]
        .into_iter()
        .map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    assert_eq!(ckpt.list_by_role(Role::Param).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Buffer).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Optimizer).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Rng).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::User).len(), 1);

    Ok(())
}
