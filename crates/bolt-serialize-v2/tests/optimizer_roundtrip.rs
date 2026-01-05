use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

type B = CpuBackend;
type D = f32;

/// Test: Save and load optimizer with velocity state.
/// Expected: Velocity state is preserved, allowing training to resume with momentum.
#[test]
fn optimizer_with_velocity_state_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("optimizer_roundtrip");

    // Create store and optimizer
    let store = Store::<B, D>::new(backend.clone(), 100);
    let w = store.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 2.0],
        &[2],
    )?)?;
    store.seal();

    let mut optim = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    // Take a step to create velocity state
    w.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.5, 0.5],
        &[2],
    )?));
    optim.step(&store.trainable())?;

    // Verify velocity state exists
    assert!(optim.velocity_state().contains_key("weight"));
    let velocity_before: Vec<f32> = optim.velocity_state()["weight"].to_vec()?;
    assert_eq!(velocity_before.len(), 2);

    // Save checkpoint
    save(
        &optim,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Create new optimizer (same config)
    let mut optim2 = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    // Load checkpoint - but optimizer needs velocity state to infer backend
    // So we need to take a step first or provide backend somehow
    // Actually, the load will fail if there's no velocity state. Let's test that first.

    // Actually, we need to have at least one parameter in velocity state to load
    // Let's create a store with the same parameter name
    let store2 = Store::<B, D>::new(backend.clone(), 200);
    let w2 = store2.param("weight", &[2], Init::Zeros)?;
    store2.seal();

    // Take a step to create velocity state
    w2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.1],
        &[2],
    )?));
    optim2.step(&store2.trainable())?;

    // Now load should work
    load(&mut optim2, &ckpt_dir, &LoadOpts::default())?;

    // Verify velocity state matches
    assert!(optim2.velocity_state().contains_key("weight"));
    let velocity_after: Vec<f32> = optim2.velocity_state()["weight"].to_vec()?;
    assert_eq!(velocity_before, velocity_after);

    Ok(())
}

/// Test: Load optimizer checkpoint when optimizer has no velocity state yet.
/// Expected: Error message explains that optimizer must be used before loading.
#[test]
fn optimizer_load_without_velocity_state_fails_gracefully() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("optimizer_no_vel");

    // Create and save optimizer with velocity
    let store = Store::<B, D>::new(backend.clone(), 1);
    let w = store.param("weight", &[2], Init::Zeros)?;
    store.seal();

    let mut optim = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    w.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 1.0],
        &[2],
    )?));
    optim.step(&store.trainable())?;

    save(
        &optim,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Try to load into fresh optimizer (no velocity state)
    let mut optim2 = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    let result = load(&mut optim2, &ckpt_dir, &LoadOpts::default());

    // Should fail with helpful error message
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("no velocity state") || err_msg.contains("infer backend"),
        "Error message should explain the issue: {}",
        err_msg
    );

    Ok(())
}

/// Test: Optimizer with multiple parameters preserves all velocity states.
/// Expected: All parameter velocity states are saved and loaded correctly.
#[test]
fn optimizer_multiple_params_velocity_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("optimizer_multi");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let w1 = store.param("layer1.weight", &[2], Init::Zeros)?;
    let w2 = store.param("layer2.weight", &[2], Init::Zeros)?;
    store.seal();

    let mut optim = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    w1.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0, 1.0],
        &[2],
    )?));
    w2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[2.0, 2.0],
        &[2],
    )?));
    optim.step(&store.trainable())?;

    let velocities_before: std::collections::BTreeMap<String, Vec<f32>> = optim
        .velocity_state()
        .iter()
        .map(|(k, v)| (k.clone(), v.to_vec().unwrap()))
        .collect();

    save(
        &optim,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Create new store and optimizer
    let store2 = Store::<B, D>::new(backend.clone(), 2);
    let w1_2 = store2.param("layer1.weight", &[2], Init::Zeros)?;
    let w2_2 = store2.param("layer2.weight", &[2], Init::Zeros)?;
    store2.seal();

    let mut optim2 = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    w1_2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.1],
        &[2],
    )?));
    w2_2.set_grad(Some(bolt_tensor::Tensor::from_slice(
        &backend,
        &[0.1, 0.1],
        &[2],
    )?));
    optim2.step(&store2.trainable())?;

    load(&mut optim2, &ckpt_dir, &LoadOpts::default())?;

    let velocities_after: std::collections::BTreeMap<String, Vec<f32>> = optim2
        .velocity_state()
        .iter()
        .map(|(k, v)| (k.clone(), v.to_vec().unwrap()))
        .collect();

    assert_eq!(velocities_before, velocities_after);

    Ok(())
}
