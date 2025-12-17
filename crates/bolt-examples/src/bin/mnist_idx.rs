//! MNIST training example using bolt-datasets.
//!
//! Demonstrates:
//! - Using `bolt_datasets::mnist` for download + streaming
//! - Training a simple MLP on CPU
//! - (Optional) Injecting augmentations via `Stream::enumerate().map(...)` for reproducible experiments

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use bolt_autodiff::AutodiffTensorExt;
use bolt_cpu::CpuBackend;
use bolt_datasets::mnist::{self, INPUT_DIM, MnistBatch, NUM_CLASSES};
use bolt_losses::{Reduction, cross_entropy_from_logits, metrics::accuracy_top1};
use bolt_nn::init::Init;
use bolt_nn::layers::{ModelExt, linear, relu};
use bolt_nn::{Context, HasParams, Model};
use bolt_optim::Sgd;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct ExperimentConfig {
    seed: u64,
    batch_size: usize,
    max_steps: usize,
    shuffle_buffer: usize,
}

fn run_experiment(
    cfg: &ExperimentConfig,
    data_root: &PathBuf,
    backend: &Arc<CpuBackend>,
) -> Result<(), Box<dyn Error>> {
    type B = CpuBackend;
    type D = f32;

    let hidden_dim = 128;

    let layer1 = linear(INPUT_DIM, hidden_dim)
        .with_weight_init(Init::KaimingUniform {
            a: (5.0f32).sqrt(),
            mode: bolt_nn::init::FanMode::FanIn,
            nonlinearity: bolt_nn::init::Nonlinearity::ReLU,
        })
        .with_bias_init(Init::Zeros)
        .build(backend)?;

    let layer2 = linear(hidden_dim, NUM_CLASSES)
        .with_weight_init(Init::KaimingUniform {
            a: (5.0f32).sqrt(),
            mode: bolt_nn::init::FanMode::FanIn,
            nonlinearity: bolt_nn::init::Nonlinearity::Linear,
        })
        .with_bias_init(Init::Zeros)
        .build(backend)?;

    let mut model = layer1.then(relu()).then(layer2);

    let mut optimizer = Sgd::<B, D>::builder()
        .learning_rate(0.1)
        .init(&model.params())?;

    let rng = StdRng::seed_from_u64(cfg.seed);
    let base_stream = mnist::train(data_root)?;

    let train_stream = base_stream
        .shuffle(cfg.shuffle_buffer, rng)
        .batch(cfg.batch_size)
        .to_batches::<B, MnistBatch<B>>(backend.clone())
        .take(cfg.max_steps);

    println!("Training for {} steps ...", cfg.max_steps);

    for (step, batch_res) in train_stream.iter().enumerate() {
        let batch = batch_res?;
        let ctx = Context::grad(backend);

        let logits = model.forward(&ctx, ctx.input(&batch.images))?;
        let targets = ctx.input(&batch.targets);
        let loss = cross_entropy_from_logits(&logits, &targets, Reduction::Mean)?;

        ctx.backward(&loss, &mut model.params_mut())?;
        optimizer.step(&mut model.params_mut())?;
        model.zero_grad();

        if (step + 1) % 50 == 0 || step == 0 {
            let acc = accuracy_top1(&logits.detach(), &batch.labels)?;
            let loss_val = loss.to_vec()?.get(0).copied().unwrap_or(0.0);
            println!(
                "  step {:>4} | step-loss = {:>8.4} | step-acc = {:>5.1}%",
                step + 1,
                loss_val,
                acc * 100.0
            );
        }
    }

    let test_stream = mnist::test(data_root)?
        .batch(256)
        .to_batches::<B, MnistBatch<B>>(backend.clone())
        .take(1);

    let eval_ctx = Context::eval(backend);
    if let Some(batch_res) = test_stream.iter().next() {
        let batch = batch_res?;
        let logits = model.forward(&eval_ctx, eval_ctx.input(&batch.images))?;
        let acc = accuracy_top1(&logits, &batch.labels)?;
        println!("  Test accuracy (first 256): {:.2}%", acc * 100.0);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());
    let data_root = PathBuf::from(data_dir);

    println!("Using MNIST data directory: {:?}", data_root);
    mnist::ensure_downloaded(&data_root)?;

    let backend = Arc::new(CpuBackend::new());

    println!("\n=== Baseline ===");
    run_experiment(
        &ExperimentConfig {
            seed: 42,
            batch_size: 128,
            max_steps: 200,
            shuffle_buffer: 10_000,
        },
        &data_root,
        &backend,
    )?;

    Ok(())
}
