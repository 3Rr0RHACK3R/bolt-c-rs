//! MNIST training example using bolt-datasets.
//!
//! Demonstrates:
//! - Using functional data pipeline with collate
//! - Training a simple MLP on CPU

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use bolt_autodiff::AutodiffTensorExt;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_datasets::mnist::{self, INPUT_DIM, NUM_CLASSES};
use bolt_losses::{Reduction, cross_entropy_from_logits, metrics::accuracy_top1};
use bolt_nn::init::Init;
use bolt_nn::layers::{ModelExt, flatten, linear, relu};
use bolt_nn::{Context, HasParams, Model};
use bolt_optim::Sgd;
use bolt_vision::{ops::tensor, types::ImageLayout};
use rand::SeedableRng;
use rand::rngs::StdRng;

const MEAN: [f32; 1] = [0.1307];
const STD: [f32; 1] = [0.3081];

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

    let mut model = flatten(1).then(layer1).then(relu()).then(layer2);

    let mut optimizer = Sgd::<B, D>::builder()
        .learning_rate(0.1)
        .init(&model.params())?;

    let rng = StdRng::seed_from_u64(cfg.seed);
    let base_stream = mnist::train(data_root)?;

    let train_stream = base_stream
        .map_with(backend.clone(), |b, ex| mnist::to_tensor_label(b, ex))
        .try_map(|mut s| {
            s.image = tensor::scale(1.0 / 255.0, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .try_map(|mut s| {
            s.image = tensor::normalize(&MEAN, &STD, ImageLayout::NCHW, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .shuffle(cfg.shuffle_buffer, rng)
        .batch(cfg.batch_size)
        .try_map_with(backend.clone(), |b, xs| {
            let images = Tensor::stack(b, xs.iter().map(|s| &s.image))?;
            let labels = Tensor::from_iter(b, xs.iter().map(|s| s.label))?;
            Ok::<_, bolt_data::DataError>(mnist::MnistBatch { images, labels })
        })
        .take(cfg.max_steps);

    println!("Training for {} steps ...", cfg.max_steps);

    for (step, batch_res) in train_stream.iter().enumerate() {
        let batch = batch_res?;
        let ctx = Context::grad(backend);

        let logits = model.forward(&ctx, ctx.input(&batch.images))?;

        let labels_vec = batch.labels.to_vec()?;
        let mut targets_buf = vec![0.0f32; labels_vec.len() * NUM_CLASSES];
        for (i, &label) in labels_vec.iter().enumerate() {
            targets_buf[i * NUM_CLASSES + label as usize] = 1.0;
        }
        let targets = ctx.input(&bolt_core::Tensor::from_slice(
            backend,
            &targets_buf,
            &[labels_vec.len(), NUM_CLASSES],
        )?);
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
        .map_with(backend.clone(), |b, ex| mnist::to_tensor_label(b, ex))
        .try_map(|mut s| {
            s.image = tensor::scale(1.0 / 255.0, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .try_map(|mut s| {
            s.image = tensor::normalize(&MEAN, &STD, ImageLayout::NCHW, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .batch(256)
        .try_map_with(backend.clone(), |b, xs| {
            let images = Tensor::stack(b, xs.iter().map(|s| &s.image))?;
            let labels = Tensor::from_iter(b, xs.iter().map(|s| s.label))?;
            Ok::<_, bolt_data::DataError>(mnist::MnistBatch { images, labels })
        })
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
