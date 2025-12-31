use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::Store;
use bolt_serialize::{
    CheckpointMeta, LoadOpts, RestoreOpts, SaveOpts, StoreCheckpointAdapter, load_checkpoint,
    save_checkpoint,
};
use bolt_tensor::Tensor;

type B = CpuBackend;

#[test]
fn restore_modelstore_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("restore_roundtrip");

    let store_src = Store::<B, f32>::new(backend.clone(), 123);
    let p = store_src.param("w", &[2], bolt_nn::Init::Zeros)?;
    p.set_tensor(Tensor::from_slice(&backend, &[10.0f32, 20.0], &[2])?)?;
    let b = store_src.buffer("buf", &[2], bolt_nn::Init::Zeros)?;
    b.set(Tensor::from_slice(&backend, &[1.0f32, 2.0], &[2])?)?;

    save_checkpoint(
        store_src.to_records(),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let store_dst = Store::<B, f32>::new(backend.clone(), 456);
    let p2 = store_dst.param("w", &[2], bolt_nn::Init::Zeros)?;
    let b2 = store_dst.buffer("buf", &[2], bolt_nn::Init::Zeros)?;

    let report = store_dst.restore_from_checkpoint(&ckpt, &RestoreOpts::default())?;
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.mismatched.is_empty());

    assert_eq!(p.tensor().to_vec()?, p2.tensor().to_vec()?);
    assert_eq!(b.tensor().to_vec()?, b2.tensor().to_vec()?);

    Ok(())
}

#[test]
fn restore_modelstore_with_rename() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("restore_rename");

    let store_src = Store::<B, f32>::new(backend.clone(), 0);
    let p = store_src
        .sub("encoder")
        .param("w", &[2], bolt_nn::Init::Zeros)?;
    p.set_tensor(Tensor::from_slice(&backend, &[3.0f32, 4.0], &[2])?)?;

    save_checkpoint(
        store_src.to_records(),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let store_dst = Store::<B, f32>::new(backend.clone(), 0);
    let p2 = store_dst
        .sub("encoder_new")
        .param("w", &[2], bolt_nn::Init::Zeros)?;

    let report = store_dst.restore_from_checkpoint(
        &ckpt,
        &RestoreOpts {
            strict: true,
            filter: None,
            rename: Some(Arc::new(|old| old.replace("encoder.", "encoder_new."))),
        },
    )?;
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.mismatched.is_empty());

    assert_eq!(p.tensor().to_vec()?, p2.tensor().to_vec()?);

    Ok(())
}
