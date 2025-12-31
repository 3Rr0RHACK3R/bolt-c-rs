use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, LoadOpts, OnError, Record, RecordMeta, Role, SaveOpts, inspect,
    load_checkpoint, save_checkpoint,
};
use tempfile::TempDir;

fn make_record(name: &str, role: Role, byte_len: usize) -> Record<'static> {
    debug_assert!(
        byte_len % 4 == 0,
        "byte length must be divisible by 4 for F32 dtype"
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

#[test]
fn checkpoint_exclude_by_pattern() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_pattern");

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
    let mut names: Vec<_> = ckpt.list().into_iter().map(|m| m.name).collect();
    names.sort();

    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(names.contains(&"encoder.bias".to_string()));
    assert!(!names.contains(&"decoder.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_glob_is_not_substring() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_glob_wildcard");

    let records = vec![
        make_record("encoder.weight", Role::Param, 100),
        make_record("decoder.weight", Role::Param, 100),
        make_record("decoder.bias", Role::Param, 100),
        make_record("encoder.decoder.weight", Role::Param, 100),
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
    assert!(!names.contains(&"decoder.weight".to_string()));
    assert!(!names.contains(&"decoder.bias".to_string()));
    assert!(names.contains(&"encoder.decoder.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_no_substring_matching() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_no_substring");

    let records = vec![
        make_record("layer1.weight", Role::Param, 100),
        make_record("layer2.weight", Role::Param, 100),
        make_record("embedding_layer.weight", Role::Param, 100),
        make_record("layer.weight", Role::Param, 100),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            exclude: vec!["layer".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let names: Vec<_> = ckpt.list().into_iter().map(|m| m.name).collect();

    assert!(names.contains(&"layer1.weight".to_string()));
    assert!(names.contains(&"layer2.weight".to_string()));
    assert!(names.contains(&"embedding_layer.weight".to_string()));
    assert!(names.contains(&"layer.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_multiple_patterns() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_multiple");

    let records = vec![
        make_record("encoder.weight", Role::Param, 100),
        make_record("decoder.weight", Role::Param, 100),
        make_record("head.weight", Role::Param, 100),
        make_record("model.tmp_buffer", Role::Buffer, 100),
    ];

    save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            exclude: vec!["decoder.*".to_string(), "*.tmp*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let names: Vec<_> = ckpt.list().into_iter().map(|m| m.name).collect();

    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(names.contains(&"head.weight".to_string()));
    assert!(!names.contains(&"decoder.weight".to_string()));
    assert!(!names.contains(&"model.tmp_buffer".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_invalid_pattern() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_invalid");

    let records = vec![make_record("model.weight", Role::Param, 100)];

    let result = save_checkpoint(
        records.into_iter().map(Ok),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            exclude: vec!["[invalid".to_string()],
            ..Default::default()
        },
    );

    match result {
        Err(bolt_serialize::Error::InvalidExcludePattern { pattern, .. }) => {
            assert_eq!(pattern, "[invalid");
            Ok(())
        }
        other => panic!("expected InvalidExcludePattern, got: {other:?}"),
    }
}

#[test]
fn checkpoint_metadata_roundtrip() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("metadata_test");

    let user_data = serde_json::json!({
        "git_commit": "abc123",
        "learning_rate": 0.001,
        "custom_field": [1, 2, 3]
    });

    let meta = CheckpointMeta {
        epoch: Some(42),
        global_step: Some(123456),
        model_name: Some("MyTestModel".to_string()),
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
    assert_eq!(ckpt.info().meta.model_name, Some("MyTestModel".to_string()));
    assert_eq!(ckpt.info().meta.user["git_commit"], "abc123");
    assert_eq!(ckpt.info().meta.user["learning_rate"], 0.001);
    assert!(!ckpt.info().written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_inspect_without_loading() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("inspect_test");

    let meta = CheckpointMeta {
        epoch: Some(99),
        global_step: Some(999999),
        model_name: Some("InspectTest".to_string()),
        ..Default::default()
    };

    save_checkpoint(
        std::iter::once(Ok(make_record("test", Role::Param, 100))),
        &out_dir,
        &meta,
        &SaveOpts::default(),
    )?;

    let info = inspect(&out_dir)?;

    assert_eq!(info.meta.epoch, Some(99));
    assert_eq!(info.meta.global_step, Some(999999));
    assert_eq!(info.meta.model_name, Some("InspectTest".to_string()));
    assert!(!info.written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_lazy_loading() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("lazy_checkpoint");

    save_checkpoint(
        std::iter::once(Ok(make_record("large", Role::Param, 10000))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(
        &out_dir,
        &LoadOpts {
            lazy: true,
            on_error: OnError::Fail,
        },
    )?;

    let view = ckpt.get("large")?;
    assert_eq!(view.shape.as_slice(), &[2500]);
    assert_eq!(view.data.len(), 10000);

    Ok(())
}

#[test]
fn checkpoint_roles_preserved_and_queryable() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("roles_preserved");

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

    let mut metas = ckpt.list();
    metas.sort_by(|a, b| a.name.cmp(&b.name));

    let role_of =
        |name: &str| -> Role { metas.iter().find(|m| m.name == name).unwrap().role.clone() };

    assert_eq!(role_of("param"), Role::Param);
    assert_eq!(role_of("buffer"), Role::Buffer);
    assert_eq!(role_of("optim"), Role::Optimizer);
    assert_eq!(role_of("rng"), Role::Rng);
    assert_eq!(role_of("user"), Role::User);

    assert_eq!(ckpt.list_by_role(Role::Param).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Buffer).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Optimizer).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::Rng).len(), 1);
    assert_eq!(ckpt.list_by_role(Role::User).len(), 1);

    Ok(())
}

#[test]
fn checkpoint_empty_meta() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("empty_meta");

    save_checkpoint(
        std::iter::once(Ok(make_record("test", Role::Param, 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    assert_eq!(ckpt.info().meta.epoch, None);
    assert_eq!(ckpt.info().meta.global_step, None);
    assert_eq!(ckpt.info().meta.model_name, None);
    assert!(!ckpt.info().written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_overwrite_behavior() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("overwrite");

    save_checkpoint(
        std::iter::once(Ok(make_record("v1", Role::Param, 100))),
        &out_dir,
        &CheckpointMeta {
            epoch: Some(1),
            ..Default::default()
        },
        &SaveOpts::default(),
    )?;

    save_checkpoint(
        std::iter::once(Ok(make_record("v2", Role::Param, 100))),
        &out_dir,
        &CheckpointMeta {
            epoch: Some(2),
            ..Default::default()
        },
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    assert_eq!(ckpt.info().meta.epoch, Some(2));
    assert!(ckpt.get("v2").is_ok());
    assert!(ckpt.get("v1").is_err());

    Ok(())
}
