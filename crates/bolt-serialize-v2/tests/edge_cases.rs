use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

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
    let w2 = store_dst.param("weight", &[3], Init::Zeros)?; // Different shape!
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
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
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

    use bolt_serialize_v2::save_all;
    save_all(
        &[
            ("model1", &store1 as &dyn bolt_serialize_v2::SaveCheckpoint),
            ("model2", &store2 as &dyn bolt_serialize_v2::SaveCheckpoint),
        ],
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load with prefix scopes
    use bolt_serialize_v2::CheckpointReader;
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

    use bolt_serialize_v2::CheckpointReader;
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
