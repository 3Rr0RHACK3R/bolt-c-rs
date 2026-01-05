use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_rng::ModelRng;
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load_all, save_all};

type B = CpuBackend;
type D = f32;

/// Test: Complete training state roundtrip (Store + Optimizer + RNG).
/// Expected: All components can be saved together and loaded to resume training exactly.
#[test]
fn complete_training_state_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("complete_training");

    // Create source training state
    let mut rng_src = ModelRng::from_seed(42);
    let store_src = Store::<B, D>::new_with_rng(backend.clone(), rng_src.init_rng());
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0],
        &[2],
    )?)?;
    store_src.seal();

    let mut optim_src = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    // Take a training step
    w.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.5, 0.5],
        &[2],
    )?));
    optim_src.step(&store_src.trainable())?;

    // Use RNG
    let _ = rng_src.forward_rngs();
    let _ = rng_src.data_rng_for_epoch(3);

    let rng_state_before = rng_src.state();
    let weight_before = w.tensor().to_vec()?;
    let velocity_before: Vec<f32> = optim_src.velocity_state()["weight"].to_vec()?;

    // Save all components with prefixes
    save_all(
        &[
            ("model", &store_src),
            ("optimizer", &optim_src),
            ("rng", &rng_src),
        ],
        &ckpt_dir,
        &CheckpointMeta {
            step: Some(100),
            epoch: Some(5),
            loss: Some(0.123),
            custom: std::collections::HashMap::new(),
        },
        &CheckpointOptions::default(),
    )?;

    // Create destination training state
    let mut rng_dst = ModelRng::from_seed(999); // Different seed
    let mut store_dst = Store::<B, D>::new_with_rng(backend.clone(), rng_dst.init_rng());
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    store_dst.seal();

    let mut optim_dst = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    // Initialize optimizer velocity state (required for loading)
    w2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.1],
        &[2],
    )?));
    optim_dst.step(&store_dst.trainable())?;

    // Load all components
    let info = load_all(
        &mut [
            ("model", &mut store_dst),
            ("optimizer", &mut optim_dst),
            ("rng", &mut rng_dst),
        ],
        &ckpt_dir,
        &LoadOpts::default(),
    )?;

    // Verify all state matches
    assert_eq!(weight_before, w2.tensor().to_vec()?);
    let velocity_after: Vec<f32> = optim_dst.velocity_state()["weight"].to_vec()?;
    assert_eq!(velocity_before, velocity_after);
    assert_eq!(rng_state_before, rng_dst.state());

    // Verify checkpoint metadata
    assert_eq!(info.meta.step, Some(100));
    assert_eq!(info.meta.epoch, Some(5));
    assert_eq!(info.meta.loss, Some(0.123));

    Ok(())
}

/// Test: Training can continue after loading checkpoint.
/// Expected: After loading, taking another step produces expected results.
#[test]
fn training_continues_after_load() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("training_continue");

    // Train for a few steps and save
    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.0, 0.0],
        &[2],
    )?)?;
    store_src.seal();

    let mut optim_src = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 1.0,       // High LR for visibility
            momentum: 0.0, // No momentum for simplicity
            weight_decay: 0.0,
        },
    )?;

    // Step 1
    w.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 1.0],
        &[2],
    )?));
    optim_src.step(&store_src.trainable())?;
    let weight_after_step1 = w.tensor().to_vec()?;

    save_all(
        &[("model", &store_src), ("optimizer", &optim_src)],
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load and continue training
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    store_dst.seal();

    let mut optim_dst = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 1.0,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )?;

    // Initialize velocity
    w2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.1],
        &[2],
    )?));
    optim_dst.step(&store_dst.trainable())?;

    // Load checkpoint
    load_all(
        &mut [("model", &mut store_dst), ("optimizer", &mut optim_dst)],
        &ckpt_dir,
        &LoadOpts::default(),
    )?;

    // Verify state matches
    assert_eq!(weight_after_step1, w2.tensor().to_vec()?);

    // Take another step
    w2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 1.0],
        &[2],
    )?));
    optim_dst.step(&store_dst.trainable())?;

    // Weight should have decreased by LR * grad = 1.0 * 1.0 = 1.0
    let weight_after_step2 = w2.tensor().to_vec()?;
    let expected: Vec<f32> = weight_after_step1.iter().map(|x| x - 1.0).collect();
    assert_eq!(weight_after_step2, expected);

    Ok(())
}
