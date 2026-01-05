use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

type B = CpuBackend;
type D = f32;

#[test]
fn store_with_params_and_buffers_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("store_roundtrip");

    // Create source store with parameters and buffers
    let store_src = Store::<B, D>::new(backend.clone(), 42);
    let w = store_src.param("weight", &[3, 2], Init::Zeros)?;
    let b = store_src.param("bias", &[2], Init::Ones)?;
    let bn_mean = store_src.buffer("bn.mean", &[2], Init::Zeros)?;
    let bn_var = store_src.buffer("bn.var", &[2], Init::Ones)?;

    // Set specific values
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    )?)?;
    b.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.5, 1.5],
        &[2],
    )?)?;
    bn_mean.set(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.2],
        &[2],
    )?)?;
    bn_var.set(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.9, 0.8],
        &[2],
    )?)?;

    store_src.seal();

    // Save checkpoint
    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Create destination store with same structure
    let mut store_dst = Store::<B, D>::new(backend.clone(), 999);
    let w2 = store_dst.param("weight", &[3, 2], Init::Zeros)?;
    let b2 = store_dst.param("bias", &[2], Init::Ones)?;
    let bn_mean2 = store_dst.buffer("bn.mean", &[2], Init::Zeros)?;
    let bn_var2 = store_dst.buffer("bn.var", &[2], Init::Ones)?;
    store_dst.seal();

    // Load checkpoint
    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify all values match
    assert_eq!(w.tensor().to_vec()?, w2.tensor().to_vec()?);
    assert_eq!(b.tensor().to_vec()?, b2.tensor().to_vec()?);
    assert_eq!(bn_mean.tensor().to_vec()?, bn_mean2.tensor().to_vec()?);
    assert_eq!(bn_var.tensor().to_vec()?, bn_var2.tensor().to_vec()?);

    Ok(())
}

#[test]
fn store_partial_load_keeps_initial_values() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("partial_load");

    // Save checkpoint with only one parameter
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

    // Load into store with additional parameter
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    let b2 = store_dst.param("bias", &[2], Init::Ones)?; // Not in checkpoint
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Weight should be loaded, bias should keep initial value
    assert_eq!(w2.tensor().to_vec()?, vec![10.0, 20.0]);
    assert_eq!(b2.tensor().to_vec()?, vec![1.0, 1.0]); // Still ones

    Ok(())
}

#[test]
fn store_empty_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("empty_store");

    let store_src = Store::<B, D>::new(backend.clone(), 1);
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Both stores should have no parameters
    assert_eq!(store_src.named_trainable().len(), 0);
    assert_eq!(store_dst.named_trainable().len(), 0);

    Ok(())
}
