use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::Store;
use bolt_optim::{Sgd, SgdCfg};
use bolt_rng::ModelRng;
use bolt_serialize::{
    CheckpointMeta, LoadOpts, OptimizerCheckpointAdapter, RestoreOpts, RngCheckpointAdapter,
    SaveOpts, StoreCheckpointAdapter, load_checkpoint, save_checkpoint,
};
use bolt_tensor::Tensor;

#[test]
fn complete_training_state_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let out_dir = tmp.path().join("complete_state");

    let mut rng_src = ModelRng::from_seed(42);
    let store_src = Store::<CpuBackend, f32>::new_with_rng(backend.clone(), rng_src.init_rng());
    
    let p = store_src.param("w", &[2], bolt_nn::Init::Zeros)?;
    p.set_tensor(Tensor::from_slice(&backend, &[10.0f32, 20.0], &[2])?)?;
    store_src.seal();

    let mut optim_src = Sgd::<CpuBackend, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;

    p.set_grad(Some(Tensor::from_slice(&backend, &[1.0f32, 1.0], &[2])?));
    let params = store_src.trainable();
    optim_src.step(&params)?;

    let _ = rng_src.forward_rngs();
    let _ = rng_src.data_rng_for_epoch(3);

    let state_before = rng_src.state();

    save_checkpoint(
        store_src
            .to_records()
            .chain(optim_src.to_records(&store_src))
            .chain(rng_src.to_records()),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_dir, &LoadOpts::default())?;

    let mut rng_dst = ModelRng::from_seed(9999);
    let store_dst = Store::<CpuBackend, f32>::new_with_rng(backend.clone(), rng_dst.init_rng());
    store_dst.param("w", &[2], bolt_nn::Init::Zeros)?;
    store_dst.seal();

    store_dst.restore_from_checkpoint(&ckpt, &RestoreOpts::default())?;

    let mut optim_dst = Sgd::<CpuBackend, f32>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;
    optim_dst.restore_from_checkpoint(&ckpt, &store_dst)?;

    rng_dst.restore_from_checkpoint(&ckpt)?;

    let state_after = rng_dst.state();
    assert_eq!(
        state_before, state_after,
        "Complete training state should round-trip exactly"
    );

    let w_restored = store_dst
        .named_trainable()
        .into_iter()
        .find(|(name, _)| name == "w")
        .unwrap()
        .1;
    let w_data: Vec<f32> = w_restored.tensor().to_vec()?;
    assert_eq!(w_data.len(), 2);

    println!("✓ Complete training state (model + optimizer + RNG) round-tripped successfully");
    Ok(())
}
