use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::Store;
use bolt_optim::{Sgd, SgdCfg};
use bolt_serialize::{
    CheckpointMeta, LoadOpts, OptimizerCheckpointAdapter, RestoreOpts, SaveOpts,
    StoreCheckpointAdapter, load_checkpoint, save_checkpoint,
};
use bolt_tensor::Tensor;

type B = CpuBackend;

fn records_by_name(
    records: impl IntoIterator<Item = bolt_serialize::Record<'static>>,
) -> BTreeMap<String, Vec<u8>> {
    records
        .into_iter()
        .map(|r| (r.meta.name, r.data.into_owned()))
        .collect()
}

#[test]
fn sgd_state_roundtrip_via_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("sgd_state_roundtrip");

    let store_src = Store::<B, f32>::new(backend.clone(), 123);
    let p = store_src.param("w", &[2], bolt_nn::Init::Zeros)?;
    p.set_tensor(Tensor::from_slice(&backend, &[10.0f32, 20.0], &[2])?)?;

    let mut optim_src = Sgd::<B, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;

    p.set_grad(Some(Tensor::from_slice(&backend, &[1.0f32, 1.0], &[2])?));
    let params = store_src.trainable();
    optim_src.step(&params)?;

    let optim_records_src: Vec<_> = optim_src
        .to_records(&store_src)
        .collect::<bolt_serialize::Result<Vec<_>>>()?;

    save_checkpoint(
        store_src
            .to_records()
            .chain(optim_src.to_records(&store_src)),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let store_dst = Store::<B, f32>::new(backend.clone(), 456);
    store_dst.param("w", &[2], bolt_nn::Init::Zeros)?;

    store_dst.restore_from_checkpoint(&ckpt, &RestoreOpts::default())?;

    let mut optim_dst = Sgd::<B, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;
    optim_dst.restore_from_checkpoint(&ckpt, &store_dst)?;

    let optim_records_dst: Vec<_> = optim_dst
        .to_records(&store_dst)
        .collect::<bolt_serialize::Result<Vec<_>>>()?;

    assert_eq!(
        records_by_name(optim_records_src),
        records_by_name(optim_records_dst)
    );

    Ok(())
}

#[test]
fn sgd_restore_errors_on_partial_velocity_state() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("sgd_partial_state");

    let store = Store::<B, f32>::new(backend.clone(), 123);
    let w1 = store.param("w1", &[2], bolt_nn::Init::Zeros)?;
    w1.set_tensor(Tensor::from_slice(&backend, &[10.0f32, 20.0], &[2])?)?;
    let w2 = store.param("w2", &[2], bolt_nn::Init::Zeros)?;
    w2.set_tensor(Tensor::from_slice(&backend, &[30.0f32, 40.0], &[2])?)?;

    let mut optim = Sgd::<B, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;

    w1.set_grad(Some(Tensor::from_slice(&backend, &[1.0f32, 1.0], &[2])?));
    w2.set_grad(Some(Tensor::from_slice(&backend, &[1.0f32, 1.0], &[2])?));
    optim.step(&store.trainable())?;

    save_checkpoint(
        store.to_records().chain(optim.to_records(&store)),
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
    manifest
        .get_mut("tensors")
        .and_then(|t| t.as_object_mut())
        .expect("manifest.tensors must be an object")
        .remove("optim.w2.vel");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let store_dst = Store::<B, f32>::new(backend.clone(), 456);
    store_dst.param("w1", &[2], bolt_nn::Init::Zeros)?;
    store_dst.param("w2", &[2], bolt_nn::Init::Zeros)?;

    let mut optim_dst = Sgd::<B, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;

    let err = optim_dst
        .restore_from_checkpoint(&ckpt, &store_dst)
        .expect_err("expected restore to fail due to missing optim.w2.vel");
    let msg = err.to_string();
    assert!(msg.contains("missing velocity"), "unexpected error: {msg}");

    Ok(())
}
