use std::fs;
use std::path::Path;
use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize_v2::{
    CheckpointMeta, CheckpointOptions, CheckpointReader, CheckpointWriter, LoadOpts, load,
    save,
};

type B = CpuBackend;
type D = f32;

/// Test: Loading from a non-existent directory fails with clear error.
/// Expected: Error message indicates the directory was not found.
#[test]
fn directory_not_found_on_load() {
    let result =
        CheckpointReader::open(Path::new("/nonexistent/path/that/does/not/exist"), &LoadOpts::default());

    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error, got Ok"),
    };
    // Should indicate that the checkpoint is invalid or not found
    assert!(
        err_msg.contains("Failed to read manifest")
            || err_msg.contains("not found")
            || err_msg.contains("InvalidCheckpoint"),
        "Error should indicate directory/manifest not found: {}",
        err_msg
    );
}

/// Test: Loading from a directory without a manifest file fails.
/// Expected: Error message indicates the manifest was not found.
#[test]
fn manifest_not_found_on_load() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let empty_dir = tmp.path().join("empty");
    fs::create_dir(&empty_dir)?;

    let result = CheckpointReader::open(&empty_dir, &LoadOpts::default());

    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error, got Ok"),
    };
    assert!(
        err_msg.contains("Failed to read manifest") || err_msg.contains("not found"),
        "Error should indicate manifest not found: {}",
        err_msg
    );

    Ok(())
}

/// Test: Loading a checkpoint with malformed JSON manifest fails.
/// Expected: Error message indicates manifest parse failure.
#[test]
fn manifest_parse_error() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let dir = tmp.path().join("bad_manifest");
    fs::create_dir_all(&dir)?;

    // Write invalid JSON
    fs::write(dir.join("bolt-checkpoint.json"), "{ invalid json }")?;

    let result = CheckpointReader::open(&dir, &LoadOpts::default());

    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error, got Ok"),
    };
    assert!(
        err_msg.contains("Failed to parse manifest") || err_msg.contains("parse"),
        "Error should indicate manifest parse failure: {}",
        err_msg
    );

    Ok(())
}

/// Test: Writing duplicate keys fails.
/// Expected: Error message indicates duplicate key.
#[test]
fn reject_duplicate_keys() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("duplicate_keys");

    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;

    let t1 = bolt_tensor::Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[2])?;
    let t2 = bolt_tensor::Tensor::<B, D>::from_slice(&backend, &[3.0, 4.0], &[2])?;

    // First write should succeed
    writer.tensor("same_key", &t1)?;

    // Second write with same key should fail
    let result = writer.tensor("same_key", &t2);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("Duplicate key") || err_msg.contains("same_key"),
        "Error should indicate duplicate key: {}",
        err_msg
    );

    Ok(())
}

/// Test: Duplicate keys within nested prefix scopes are detected.
/// Expected: Error when same full key path is written twice.
#[test]
fn reject_duplicate_keys_with_prefix() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("duplicate_prefix");

    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;

    let t = bolt_tensor::Tensor::<B, D>::from_slice(&backend, &[1.0], &[1])?;

    // Write with prefix
    writer.with_prefix("model", |w| w.tensor("weight", &t))?;

    // Try to write again with same full key path
    writer.with_prefix("model", |w| {
        let result = w.tensor("weight", &t);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Duplicate key") || err_msg.contains("model.weight"),
            "Error should indicate duplicate key: {}",
            err_msg
        );
        Ok(())
    })?;

    Ok(())
}

/// Test: Reading a non-existent key fails with clear error.
/// Expected: Error message indicates key was not found.
#[test]
fn key_not_found_on_read() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("key_not_found");

    // Save a checkpoint with one key
    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("weight", &[2], Init::Zeros)?;
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Try to read a non-existent key
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let result: Result<bolt_tensor::Tensor<B, D>, _> =
        reader.tensor("nonexistent_key", &backend);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("Key not found"),
        "Error should indicate key not found: {}",
        err_msg
    );

    Ok(())
}

/// Test: Manifest missing required fields fails gracefully.
/// Expected: Error indicates invalid checkpoint format.
#[test]
fn manifest_missing_required_fields() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let dir = tmp.path().join("incomplete_manifest");
    fs::create_dir_all(&dir)?;

    // Write incomplete but valid JSON
    fs::write(dir.join("bolt-checkpoint.json"), r#"{"unexpected": "field"}"#)?;

    let result = CheckpointReader::open(&dir, &LoadOpts::default());

    // Should fail because manifest is missing required fields
    assert!(result.is_err());

    Ok(())
}

/// Test: Valid record names are accepted.
/// Expected: Various valid naming patterns work correctly.
#[test]
fn accept_valid_record_names() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());

    let valid_names = vec![
        "layer1.weight",
        "encoder_block_0_attention_query",
        "model.layers.0.self_attn.q_proj.weight",
        "simple",
        "a",
        "tensor_123",
        "foo..bar",
    ];

    for name in valid_names {
        let tmp = tempfile::tempdir()?;
        let ckpt_dir = tmp.path().join(format!("valid_{}", name.replace('.', "_")));

        let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;
        let t = bolt_tensor::Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[2])?;
        writer.tensor(name, &t)?;
        writer.finish(&CheckpointMeta::default())?;

        // Verify we can read it back
        let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
        let loaded: bolt_tensor::Tensor<B, D> = reader.tensor(name, &backend)?;
        assert_eq!(loaded.to_vec()?, vec![1.0, 2.0]);
    }

    Ok(())
}

/// Test: Saving when parent directory doesn't exist creates it.
/// Expected: Checkpoint is saved successfully, creating necessary directories.
#[test]
fn creates_parent_directories() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("deeply/nested/checkpoint/dir");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("weight", &[2], Init::Zeros)?;
    store.seal();

    // Should create all parent directories
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    assert!(ckpt_dir.exists());
    assert!(ckpt_dir.join("bolt-checkpoint.json").exists());

    Ok(())
}

/// Test: Checkpoint can be read immediately after writing without issues.
/// Expected: No file handle or locking issues.
#[test]
fn read_immediately_after_write() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("immediate_read");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let w = store.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[42.0, 43.0],
        &[2],
    )?)?;
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Immediately read back
    let mut store2 = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store2.param("weight", &[2], Init::Zeros)?;
    store2.seal();

    load(&mut store2, &ckpt_dir, &LoadOpts::default())?;

    assert_eq!(w2.tensor().to_vec()?, vec![42.0, 43.0]);

    Ok(())
}
