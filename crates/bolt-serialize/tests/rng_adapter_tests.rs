use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::Store;
use bolt_rng::ModelRng;
use bolt_serialize::{
    CheckpointMeta, LoadOpts, RngCheckpointAdapter, SaveOpts, StoreCheckpointAdapter,
    load_checkpoint, save_checkpoint,
};

#[test]
fn rng_state_roundtrip_via_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let out_dir = tmp.path().join("rng_state_roundtrip");

    let store = Store::<CpuBackend, f32>::new(backend, 123);
    let mut rng_src = ModelRng::from_seed(42);

    let _ = rng_src.init_rng();
    let _ = rng_src.forward_rngs();
    let _ = rng_src.data_rng_for_epoch(5);

    let state_before = rng_src.state();

    save_checkpoint(
        store.to_records().chain(rng_src.to_records()),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let mut rng_dst = ModelRng::from_seed(9999);
    rng_dst.restore_from_checkpoint(&ckpt)?;

    let state_after = rng_dst.state();

    assert_eq!(state_before, state_after, "RNG state should round-trip exactly");

    Ok(())
}

#[test]
fn rng_state_produces_same_values_after_restore() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let out_dir = tmp.path().join("rng_same_values");

    let store = Store::<CpuBackend, f32>::new(backend, 123);
    let mut rng_src = ModelRng::from_seed(1337);

    let _ = rng_src.init_rng();
    let _ = rng_src.forward_rngs();

    save_checkpoint(
        store.to_records().chain(rng_src.to_records()),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let expected_init_rng = rng_src.init_rng();
    let expected_forward_rngs = rng_src.forward_rngs();

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let mut rng_dst = ModelRng::from_seed(9999);
    rng_dst.restore_from_checkpoint(&ckpt)?;

    let restored_init_rng = rng_dst.init_rng();
    let restored_forward_rngs = rng_dst.forward_rngs();

    assert_eq!(
        expected_init_rng.state(),
        restored_init_rng.state(),
        "init RNG should produce same stream after restore"
    );
    assert_eq!(
        expected_forward_rngs.state(),
        restored_forward_rngs.state(),
        "forward RNGs should produce same stream after restore"
    );

    Ok(())
}

#[test]
fn rng_restore_fails_on_missing_key() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let out_dir = tmp.path().join("rng_missing_key");

    let store = Store::<CpuBackend, f32>::new(backend, 123);
    let rng_src = ModelRng::from_seed(42);

    save_checkpoint(
        store.to_records().chain(rng_src.to_records()),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let manifest_path = out_dir.join("bolt-checkpoint.json");
    let manifest_raw = fs::read_to_string(&manifest_path)?;
    let mut manifest: serde_json::Value = serde_json::from_str(&manifest_raw)?;

    if let Some(tensors) = manifest.get_mut("tensors").and_then(|t| t.as_object_mut()) {
        tensors.remove("rng.forward.key");
    }

    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)? + "\n")?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;
    let mut rng_dst = ModelRng::from_seed(9999);
    let err = rng_dst
        .restore_from_checkpoint(&ckpt)
        .expect_err("expected restore to fail due to missing rng.forward.key");

    let msg = err.to_string();
    assert!(
        msg.contains("rng.forward.key"),
        "unexpected error: {msg}"
    );

    Ok(())
}
