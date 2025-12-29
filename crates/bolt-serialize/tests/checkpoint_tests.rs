use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointLoadOptions, CheckpointMetadata, CheckpointSaveOptions, ErrorMode, TensorMeta,
    TensorRole, TensorSetSaveOptions, TensorToSave, inspect_checkpoint, load_checkpoint,
    save_checkpoint,
};
use tempfile::TempDir;

fn make_tensor<'a>(name: &str, role: TensorRole, byte_len: usize) -> TensorToSave<'a> {
    debug_assert!(
        byte_len % 4 == 0,
        "tensor byte length must be divisible by 4 for F32 dtype"
    );
    TensorToSave::new(
        TensorMeta::new(
            name,
            DType::F32,
            Shape::from_slice(&[byte_len / 4]).unwrap(),
        )
        .with_role(role),
        vec![0u8; byte_len],
    )
}

#[test]
fn checkpoint_filters_by_role() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("filter_test");

    let tensors = vec![
        make_tensor("model.weight", TensorRole::ModelParam, 100),
        make_tensor("model.running_mean", TensorRole::ModelBuffer, 100),
        make_tensor("optimizer.momentum", TensorRole::OptimizerState, 100),
        make_tensor("rng.state", TensorRole::RngState, 100),
    ];

    save_checkpoint(
        tensors.clone(),
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            include_optimizer: false,
            include_rng: true,
            include_buffers: true,
            tensor_set: TensorSetSaveOptions {
                overwrite: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"model.weight".to_string()));
    assert!(names.contains(&"model.running_mean".to_string()));
    assert!(!names.contains(&"optimizer.momentum".to_string()));
    assert!(names.contains(&"rng.state".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_rng() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_rng");

    let tensors = vec![
        make_tensor("model.weight", TensorRole::ModelParam, 100),
        make_tensor("rng.state", TensorRole::RngState, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            include_optimizer: true,
            include_rng: false,
            include_buffers: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"model.weight".to_string()));
    assert!(!names.contains(&"rng.state".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_buffers() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_buffers");

    let tensors = vec![
        make_tensor("model.weight", TensorRole::ModelParam, 100),
        make_tensor("model.running_mean", TensorRole::ModelBuffer, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            include_optimizer: true,
            include_rng: true,
            include_buffers: false,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"model.weight".to_string()));
    assert!(!names.contains(&"model.running_mean".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_by_pattern() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_pattern");

    let tensors = vec![
        make_tensor("encoder.weight", TensorRole::ModelParam, 100),
        make_tensor("decoder.weight", TensorRole::ModelParam, 100),
        make_tensor("encoder.bias", TensorRole::ModelParam, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["decoder.*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(names.contains(&"encoder.bias".to_string()));
    assert!(!names.contains(&"decoder.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_glob_wildcard() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_glob_wildcard");

    let tensors = vec![
        make_tensor("encoder.weight", TensorRole::ModelParam, 100),
        make_tensor("decoder.weight", TensorRole::ModelParam, 100),
        make_tensor("decoder.bias", TensorRole::ModelParam, 100),
        make_tensor("encoder.decoder.weight", TensorRole::ModelParam, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["decoder.*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"encoder.weight".to_string()));
    assert!(!names.contains(&"decoder.weight".to_string()));
    assert!(!names.contains(&"decoder.bias".to_string()));
    // encoder.decoder.weight should NOT be excluded (demonstrates glob is not substring)
    assert!(names.contains(&"encoder.decoder.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_no_substring_matching() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_no_substring");

    let tensors = vec![
        make_tensor("layer1.weight", TensorRole::ModelParam, 100),
        make_tensor("layer2.weight", TensorRole::ModelParam, 100),
        make_tensor("embedding_layer.weight", TensorRole::ModelParam, 100),
        make_tensor("layer.weight", TensorRole::ModelParam, 100),
    ];

    // With glob matching, "layer" should only match exactly "layer", not substrings
    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["layer".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    // All should be included because "layer" doesn't match any of these names exactly
    assert!(names.contains(&"layer1.weight".to_string()));
    assert!(names.contains(&"layer2.weight".to_string()));
    assert!(names.contains(&"embedding_layer.weight".to_string()));
    assert!(names.contains(&"layer.weight".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_optimizer_pattern() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_optimizer");

    let tensors = vec![
        make_tensor("model.weight", TensorRole::ModelParam, 100),
        make_tensor("opt.model.weight.exp_avg", TensorRole::OptimizerState, 100),
        make_tensor("opt.model.weight.exp_avg_sq", TensorRole::OptimizerState, 100),
        make_tensor("opt.model.bias.momentum", TensorRole::OptimizerState, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["opt.*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

    assert!(names.contains(&"model.weight".to_string()));
    assert!(!names.contains(&"opt.model.weight.exp_avg".to_string()));
    assert!(!names.contains(&"opt.model.weight.exp_avg_sq".to_string()));
    assert!(!names.contains(&"opt.model.bias.momentum".to_string()));

    Ok(())
}

#[test]
fn checkpoint_exclude_multiple_patterns() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("exclude_multiple");

    let tensors = vec![
        make_tensor("encoder.weight", TensorRole::ModelParam, 100),
        make_tensor("decoder.weight", TensorRole::ModelParam, 100),
        make_tensor("head.weight", TensorRole::ModelParam, 100),
        make_tensor("model.tmp_buffer", TensorRole::ModelBuffer, 100),
    ];

    save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["decoder.*".to_string(), "*.tmp*".to_string()],
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let names: Vec<_> = ckpt.tensors.list().iter().map(|m| m.name.clone()).collect();

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

    let tensors = vec![make_tensor("model.weight", TensorRole::ModelParam, 100)];

    let result = save_checkpoint(
        tensors,
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions {
            exclude: vec!["[invalid".to_string()], // Invalid glob pattern (unclosed bracket)
            ..Default::default()
        },
    );

    assert!(result.is_err());
    match result {
        Err(bolt_serialize::Error::InvalidExcludePattern { pattern, .. }) => {
            assert_eq!(pattern, "[invalid");
        }
        _ => panic!("Expected InvalidExcludePattern error"),
    }

    Ok(())
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

    let metadata = CheckpointMetadata {
        epoch: Some(42),
        global_step: Some(123456),
        model_name: Some("MyTestModel".to_string()),
        user: user_data.clone(),
    };

    save_checkpoint(
        [make_tensor("test", TensorRole::ModelParam, 100)],
        &out_dir,
        &metadata,
        &CheckpointSaveOptions::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;

    assert_eq!(ckpt.info.metadata.epoch, Some(42));
    assert_eq!(ckpt.info.metadata.global_step, Some(123456));
    assert_eq!(ckpt.info.metadata.model_name, Some("MyTestModel".to_string()));
    assert_eq!(ckpt.info.metadata.user["git_commit"], "abc123");
    assert_eq!(ckpt.info.metadata.user["learning_rate"], 0.001);
    // Verify written_at is populated
    assert!(!ckpt.info.written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_inspect_without_loading() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("inspect_test");

    let metadata = CheckpointMetadata {
        epoch: Some(99),
        global_step: Some(999999),
        model_name: Some("InspectTest".to_string()),
        ..Default::default()
    };

    save_checkpoint(
        [make_tensor("test", TensorRole::ModelParam, 100)],
        &out_dir,
        &metadata,
        &CheckpointSaveOptions::default(),
    )?;

    let info = inspect_checkpoint(&out_dir)?;

    assert_eq!(info.metadata.epoch, Some(99));
    assert_eq!(info.metadata.global_step, Some(999999));
    assert_eq!(info.metadata.model_name, Some("InspectTest".to_string()));
    assert!(!info.written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_lazy_loading() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("lazy_checkpoint");

    save_checkpoint(
        [make_tensor("large", TensorRole::ModelParam, 10000)],
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions::default(),
    )?;

    let ckpt = load_checkpoint(
        &out_dir,
        &CheckpointLoadOptions {
            lazy: true,
            error_mode: ErrorMode::Strict,
        },
    )?;

    let view = ckpt.tensors.get("large")?;
    assert_eq!(view.shape.as_slice(), &[2500]);
    assert_eq!(view.data.len(), 10000);

    Ok(())
}

#[test]
fn checkpoint_tensor_roles_preserved() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("roles_preserved");

    save_checkpoint(
        [
            make_tensor("param", TensorRole::ModelParam, 100),
            make_tensor("buffer", TensorRole::ModelBuffer, 100),
            make_tensor("optim", TensorRole::OptimizerState, 100),
            make_tensor("rng", TensorRole::RngState, 100),
            make_tensor("user", TensorRole::User, 100),
        ],
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;
    let metas = ckpt.tensors.list();

    let find_role =
        |name: &str| -> TensorRole { metas.iter().find(|m| m.name == name).unwrap().role.clone() };

    assert_eq!(find_role("param"), TensorRole::ModelParam);
    assert_eq!(find_role("buffer"), TensorRole::ModelBuffer);
    assert_eq!(find_role("optim"), TensorRole::OptimizerState);
    assert_eq!(find_role("rng"), TensorRole::RngState);
    assert_eq!(find_role("user"), TensorRole::User);

    Ok(())
}

#[test]
fn checkpoint_empty_metadata() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("empty_meta");

    save_checkpoint(
        [make_tensor("test", TensorRole::ModelParam, 100)],
        &out_dir,
        &CheckpointMetadata::default(),
        &CheckpointSaveOptions::default(),
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;

    assert_eq!(ckpt.info.metadata.epoch, None);
    assert_eq!(ckpt.info.metadata.global_step, None);
    assert_eq!(ckpt.info.metadata.model_name, None);
    // written_at should still be set
    assert!(!ckpt.info.written_at.is_empty());

    Ok(())
}

#[test]
fn checkpoint_overwrite_behavior() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("overwrite");

    let metadata_v1 = CheckpointMetadata {
        epoch: Some(1),
        ..Default::default()
    };

    save_checkpoint(
        [make_tensor("v1", TensorRole::ModelParam, 100)],
        &out_dir,
        &metadata_v1,
        &CheckpointSaveOptions::default(),
    )?;

    let metadata_v2 = CheckpointMetadata {
        epoch: Some(2),
        ..Default::default()
    };

    save_checkpoint(
        [make_tensor("v2", TensorRole::ModelParam, 100)],
        &out_dir,
        &metadata_v2,
        &CheckpointSaveOptions {
            tensor_set: TensorSetSaveOptions {
                overwrite: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &CheckpointLoadOptions::default())?;

    assert_eq!(ckpt.info.metadata.epoch, Some(2));
    assert!(ckpt.tensors.get("v2").is_ok());
    assert!(ckpt.tensors.get("v1").is_err()); // Old tensor should be gone

    Ok(())
}
