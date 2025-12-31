#[path = "../mnist_common.rs"]
mod mnist_common;

use std::path::PathBuf;
use std::sync::Arc;

use bolt_datasets::mnist;
use bolt_losses::{Reduction, accuracy_top1, cross_entropy_from_logits_sparse};
use bolt_nn::{ForwardCtx, Module};
use bolt_optim::{Sgd, SgdCfg, SgdGroupCfg};
use bolt_rng::{RngStream, RngStreams};

use bolt_serialize::{CheckpointMeta, SaveOpts, StoreCheckpointAdapter, save_checkpoint};

use mnist_common::{B, D, MnistMLP, evaluate, train_loader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed = 42u64;
    let data_root = std::env::var("MNIST_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| std::path::PathBuf::from("data/mnist"));
    mnist::ensure_downloaded(&data_root)?;

    let backend = Arc::new(B::new());
    let store = bolt_nn::Store::<B, D>::new(backend.clone(), 1337);
    let model = MnistMLP::init(&store, 512, 256)?;
    let mut model_rngs = RngStreams::from_seed(seed);

    store.group_params_by_name(|name| name.contains("bias"), 1);
    store.seal();

    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 0.01,
        momentum: 0.9,
        weight_decay: 1e-4,
    })?;
    opt.set_group(
        1,
        SgdGroupCfg {
            lr_mult: 1.0,
            weight_decay: Some(0.0),
        },
    )?;
    let params = store.trainable();

    let batch_size: usize = std::env::var("MNIST_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let epochs: usize = std::env::var("MNIST_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    for epoch in 0..epochs {
        let rng = RngStream::from_seed(seed + epoch as u64);
        let loader = train_loader(&data_root, backend.clone(), batch_size, rng)?;

        for (step, batch_res) in loader.iter().enumerate() {
            let batch = batch_res?;
            store.zero_grad();

            let mut ctx = ForwardCtx::train_with_rngs(model_rngs.split());
            let logits = model.forward(batch.images, &mut ctx)?;
            let loss = cross_entropy_from_logits_sparse(
                &logits,
                &batch.labels,
                mnist::NUM_CLASSES,
                Reduction::Mean,
            )?;

            store.backward(&loss)?;
            opt.step(&params)?;

            if (step + 1) % 50 == 0 || step == 0 {
                let loss_val = loss.item()?;
                let logits_detached = logits.clone().with_requires_grad(false);
                let acc = accuracy_top1(&logits_detached, &batch.labels)?;
                println!(
                    "epoch {:>2} step {:>4} loss {:>8.4} acc {:>5.1}%",
                    epoch + 1,
                    step + 1,
                    loss_val,
                    acc * 100.0
                );
            }
        }

        if (epoch + 1) % 2 == 0 {
            println!("\n--- Epoch {} Test Evaluation ---", epoch + 1);
            let eval = evaluate(&model, &data_root, backend.clone(), batch_size)?;
            println!(
                "Test Loss: {:.4}  Test Acc: {:.2}%",
                eval.loss,
                eval.acc * 100.0
            );
        }
    }

    println!("\n=== Final Evaluation on Test Set ===");
    let eval = evaluate(&model, &data_root, backend, batch_size)?;
    println!(
        "Test Loss: {:.4}  Test Acc: {:.2}%",
        eval.loss,
        eval.acc * 100.0
    );

    let ckpt_dir: PathBuf = std::env::var("CKPT_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| PathBuf::from("data/mnist_ckpt"));

    save_checkpoint(
        store.to_records(),
        &ckpt_dir,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;
    println!("Saved checkpoint to {}", ckpt_dir.display());

    println!(
        "Recorded final test accuracy: {:.2}% ({}/{})",
        eval.acc * 100.0,
        eval.correct,
        eval.total
    );
    Ok(())
}
