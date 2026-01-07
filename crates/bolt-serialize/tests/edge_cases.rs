use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

type B = CpuBackend;
type D = f32;

/// Test: Loading checkpoint with shape mismatch fails gracefully.
/// Expected: Clear error message explaining the shape mismatch.
#[test]
fn load_with_shape_mismatch_fails() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("shape_mismatch");

    // Save parameter with shape [2]
    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0],
        &[2],
    )?)?;
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Try to load into parameter with different shape [3]
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let _w2 = store_dst.param("weight", &[3], Init::Zeros)?; // Different shape!
    store_dst.seal();

    let result = load(&mut store_dst, &ckpt_dir, &LoadOpts::default());

    // Should fail with shape mismatch error
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    // Error should mention shape or size mismatch
    assert!(
        err_msg.contains("shape") || err_msg.contains("size") || err_msg.contains("mismatch"),
        "Error should mention shape/size issue: {}",
        err_msg
    );

    Ok(())
}

/// Test: Loading checkpoint with dtype mismatch fails gracefully.
/// Expected: Clear error message explaining the dtype mismatch.
#[test]
fn load_with_dtype_mismatch_fails() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("dtype_mismatch");

    // Save f32 parameter
    let store_src = Store::<B, f32>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0f32, 2.0],
        &[2],
    )?)?;
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Try to load into f64 parameter (different dtype)
    let mut store_dst = Store::<B, f64>::new(backend.clone(), 2);
    let _w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    store_dst.seal();

    let result = load(&mut store_dst, &ckpt_dir, &LoadOpts::default());

    // Should fail with dtype mismatch error
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("dtype") || err_msg.contains("DType") || err_msg.contains("type"),
        "Error should mention dtype issue: {}",
        err_msg
    );

    Ok(())
}

/// Test: Scoped reader correctly filters keys by prefix.
/// Expected: Only keys with the specified prefix are accessible.
#[test]
fn scoped_reader_filters_by_prefix() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("scoped_prefix");

    // Save multiple items with prefixes
    let store1 = Store::<B, D>::new(backend.clone(), 1);
    let w1 = store1.param("weight", &[2], Init::Zeros)?;
    w1.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0],
        &[2],
    )?)?;
    store1.seal();

    let store2 = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store2.param("weight", &[2], Init::Zeros)?;
    w2.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[3.0, 4.0],
        &[2],
    )?)?;
    store2.seal();

    use bolt_serialize::save_all;
    save_all(
        &[
            ("model1", &store1 as &dyn bolt_serialize::SaveCheckpoint),
            ("model2", &store2 as &dyn bolt_serialize::SaveCheckpoint),
        ],
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load with prefix scopes
    use bolt_serialize::CheckpointReader;
    let mut reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;

    // Each prefix scope should only see its own keys
    let model1_keys: Vec<String> = reader.with_prefix("model1", |r| Ok(r.keys()))?;
    assert!(model1_keys.contains(&"weight".to_string()));
    assert!(!model1_keys.contains(&"model2.weight".to_string()));

    let model2_keys: Vec<String> = reader.with_prefix("model2", |r| Ok(r.keys()))?;
    assert!(model2_keys.contains(&"weight".to_string()));
    assert!(!model2_keys.contains(&"model1.weight".to_string()));

    // Load and verify values
    let mut store1_dst = Store::<B, D>::new(backend.clone(), 3);
    let w1_dst = store1_dst.param("weight", &[2], Init::Zeros)?;
    store1_dst.seal();
    reader.load_prefixed("model1", &mut store1_dst)?;
    assert_eq!(w1.tensor().to_vec()?, w1_dst.tensor().to_vec()?);

    let mut store2_dst = Store::<B, D>::new(backend.clone(), 4);
    let w2_dst = store2_dst.param("weight", &[2], Init::Zeros)?;
    store2_dst.seal();
    reader.load_prefixed("model2", &mut store2_dst)?;
    assert_eq!(w2.tensor().to_vec()?, w2_dst.tensor().to_vec()?);

    Ok(())
}

/// Test: Checkpoint metadata is preserved and accessible.
/// Expected: All metadata fields (step, epoch, loss, custom) are saved and loaded correctly.
#[test]
fn checkpoint_metadata_preserved() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("metadata");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("weight", &[2], Init::Zeros)?;
    store.seal();

    let mut custom = std::collections::HashMap::new();
    custom.insert("model_name".to_string(), serde_json::json!("TestModel"));
    custom.insert("version".to_string(), serde_json::json!(42));

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta {
            step: Some(1000),
            epoch: Some(10),
            loss: Some(0.5),
            custom: custom.clone(),
        },
        &CheckpointOptions::default(),
    )?;

    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    assert_eq!(info.meta.step, Some(1000));
    assert_eq!(info.meta.epoch, Some(10));
    assert_eq!(info.meta.loss, Some(0.5));
    assert_eq!(
        info.meta.custom.get("model_name"),
        Some(&serde_json::json!("TestModel"))
    );
    assert_eq!(
        info.meta.custom.get("version"),
        Some(&serde_json::json!(42))
    );

    Ok(())
}

/// Test: Saving to same directory overwrites previous checkpoint.
/// Expected: New checkpoint replaces old one completely.
#[test]
fn overwrite_behavior() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("overwrite_test");

    // Save first checkpoint
    let store_v1 = Store::<B, D>::new(backend.clone(), 1);
    let w1 = store_v1.param("v1_weight", &[2], Init::Zeros)?;
    w1.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0],
        &[2],
    )?)?;
    store_v1.seal();

    save(
        &store_v1,
        &ckpt_dir,
        &CheckpointMeta {
            epoch: Some(1),
            ..Default::default()
        },
        &CheckpointOptions::default(),
    )?;

    // Save second checkpoint to same directory
    let store_v2 = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store_v2.param("v2_weight", &[3], Init::Zeros)?;
    w2.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[10.0, 20.0, 30.0],
        &[3],
    )?)?;
    store_v2.seal();

    save(
        &store_v2,
        &ckpt_dir,
        &CheckpointMeta {
            epoch: Some(2),
            ..Default::default()
        },
        &CheckpointOptions::default(),
    )?;

    // Load and verify we get the new checkpoint
    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    // Should have epoch 2 (from new checkpoint)
    assert_eq!(info.meta.epoch, Some(2));

    // Should have v2_weight, not v1_weight
    assert!(reader.contains("v2_weight"));
    assert!(!reader.contains("v1_weight"));

    // Verify data is from new checkpoint
    let mut store_dst = Store::<B, D>::new(backend.clone(), 3);
    let w_dst = store_dst.param("v2_weight", &[3], Init::Zeros)?;
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;
    assert_eq!(w_dst.tensor().to_vec()?, vec![10.0, 20.0, 30.0]);

    Ok(())
}

/// Test: Reader.contains() returns correct results.
/// Expected: Returns true for existing keys, false for non-existent keys.
#[test]
fn reader_contains_works_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("contains_test");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("existing_weight", &[2], Init::Zeros)?;
    let _b = store.param("existing_bias", &[2], Init::Zeros)?;
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;

    // Existing keys
    assert!(reader.contains("existing_weight"));
    assert!(reader.contains("existing_bias"));

    // Non-existent keys
    assert!(!reader.contains("nonexistent"));
    assert!(!reader.contains(""));
    assert!(!reader.contains("existing"));  // Partial match should not work

    Ok(())
}

/// Test: Reader.keys() lists all keys in checkpoint.
/// Expected: Returns complete list of all saved keys.
#[test]
fn reader_keys_lists_all() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("keys_list");

    let store = Store::<B, D>::new(backend.clone(), 1);
    store.param("alpha", &[2], Init::Zeros)?;
    store.param("beta", &[2], Init::Zeros)?;
    store.param("gamma", &[2], Init::Zeros)?;
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let mut keys = reader.keys();
    keys.sort();

    assert_eq!(keys, vec!["alpha", "beta", "gamma"]);

    Ok(())
}

/// Test: Empty metadata fields are handled correctly.
/// Expected: None values are preserved as None.
#[test]
fn empty_metadata_fields_preserved() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("empty_meta");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("weight", &[2], Init::Zeros)?;
    store.seal();

    // Save with completely default (empty) metadata
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    assert_eq!(info.meta.step, None);
    assert_eq!(info.meta.epoch, None);
    assert_eq!(info.meta.loss, None);
    assert!(info.meta.custom.is_empty());

    Ok(())
}

/// Test: Loading into store with extra parameters keeps them unchanged.
/// Expected: Parameters not in checkpoint retain their initial/current values.
#[test]
fn extra_params_in_store_unchanged() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("extra_params");

    // Save checkpoint with only 'weight'
    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[10.0, 20.0],
        &[2],
    )?)?;
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load into store with extra parameter 'bias'
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let w_dst = store_dst.param("weight", &[2], Init::Zeros)?;
    let b_dst = store_dst.param("bias", &[2], Init::Ones)?; // Initialized to ones
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Weight should be loaded from checkpoint
    assert_eq!(w_dst.tensor().to_vec()?, vec![10.0, 20.0]);
    // Bias should retain initial value (ones)
    assert_eq!(b_dst.tensor().to_vec()?, vec![1.0, 1.0]);

    Ok(())
}

/// Test: Prefixed save/load with nested prefixes.
/// Expected: Nested prefixes are correctly concatenated.
#[test]
fn nested_prefix_scopes() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("nested_prefixes");

    // Create writer and use nested prefixes
    use bolt_serialize::CheckpointWriter;
    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;

    let t = bolt_tensor::Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[2])?;

    writer.with_prefix("layer1", |w| {
        w.with_prefix("attention", |w2| {
            w2.tensor("query", &t)?;
            w2.tensor("key", &t)?;
            Ok(())
        })
    })?;

    writer.with_prefix("layer2", |w| {
        w.tensor("output", &t)?;
        Ok(())
    })?;

    writer.finish(&CheckpointMeta::default())?;

    // Read and verify nested keys
    use bolt_serialize::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let keys = reader.keys();

    assert!(keys.contains(&"layer1.attention.query".to_string()));
    assert!(keys.contains(&"layer1.attention.key".to_string()));
    assert!(keys.contains(&"layer2.output".to_string()));

    // Verify data can be read
    let query: bolt_tensor::Tensor<B, D> =
        reader.tensor("layer1.attention.query", &backend)?;
    assert_eq!(query.to_vec()?, vec![1.0, 2.0]);

    Ok(())
}
