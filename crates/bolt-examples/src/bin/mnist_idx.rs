//! MNIST training example using bolt-datasets.
//!
//! Demonstrates:
//! - Using functional data pipeline with collate
//! - Training a simple MLP on CPU
//! - Custom model struct implementing Model trait directly

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use bolt_autodiff::AutodiffTensorExt;
use bolt_autodiff::HasParams;
use bolt_autodiff::Parameter;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_datasets::mnist::{self, INPUT_DIM, NUM_CLASSES};
use bolt_losses::{Reduction, cross_entropy_from_logits, metrics::accuracy_top1};
use bolt_nn::init::Init;
use bolt_nn::layers::{Linear, linear};
use bolt_nn::{Compute, ComputeOps, Context, Mode, Model, Trainable};
use bolt_optim::Sgd;
use bolt_vision::{ops::tensor, types::ImageLayout};
use rand::SeedableRng;
use rand::rngs::StdRng;

const MEAN: [f32; 1] = [0.1307];
const STD: [f32; 1] = [0.3081];

pub struct MnistMLP<B, D>
where
    B: Compute<D>,
    D: bolt_autodiff::Float,
{
    fc1: Linear<B, D>,
    fc2: Linear<B, D>,
}

impl<B, D> MnistMLP<B, D>
where
    B: Compute<D> + bolt_core::backend::RandomOp<D>,
    D: bolt_autodiff::Float,
{
    pub fn new(backend: &Arc<B>, hidden_dim: usize) -> bolt_nn::Result<Self> {
        let fc1 = linear(INPUT_DIM, hidden_dim)
            .with_weight_init(Init::KaimingUniform {
                a: (5.0f32).sqrt(),
                mode: bolt_nn::init::FanMode::FanIn,
                nonlinearity: bolt_nn::init::Nonlinearity::ReLU,
            })
            .with_bias_init(Init::Zeros)
            .build(backend)?;

        let fc2 = linear(hidden_dim, NUM_CLASSES)
            .with_weight_init(Init::KaimingUniform {
                a: (5.0f32).sqrt(),
                mode: bolt_nn::init::FanMode::FanIn,
                nonlinearity: bolt_nn::init::Nonlinearity::Linear,
            })
            .with_bias_init(Init::Zeros)
            .build(backend)?;

        Ok(Self { fc1, fc2 })
    }
}

impl<B, D> Trainable for MnistMLP<B, D>
where
    B: Compute<D>,
    D: bolt_autodiff::Float,
{
}

impl<B, D> HasParams<B, D> for MnistMLP<B, D>
where
    B: Compute<D>,
    D: bolt_autodiff::Float,
{
    fn visit_params<'a>(&'a self, f: &mut dyn FnMut(&'a Parameter<B, D>)) {
        self.fc1.visit_params(f);
        self.fc2.visit_params(f);
    }

    fn visit_params_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Parameter<B, D>)) {
        self.fc1.visit_params_mut(f);
        self.fc2.visit_params_mut(f);
    }

    fn param_count(&self) -> usize {
        self.fc1.param_count() + self.fc2.param_count()
    }
}

impl<B, D, M> Model<B, D, M> for MnistMLP<B, D>
where
    B: Compute<D>,
    D: bolt_autodiff::Float,
    M: Mode<B, D>,
    M::Backend: ComputeOps<D>,
{
    type Input = Tensor<M::Backend, D>;
    type Output = bolt_nn::Result<Tensor<M::Backend, D>>;

    fn forward(&self, ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output {
        let batch_size = input.shape()[0];
        let x = input.reshape(&[batch_size, INPUT_DIM])?;

        let x = self.fc1.forward(ctx, x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(ctx, x)?;

        Ok(x)
    }
}

struct ExperimentConfig {
    seed: u64,
    batch_size: usize,
    epochs: usize,
    shuffle_buffer: usize,
}

fn run_experiment(
    cfg: &ExperimentConfig,
    data_root: &PathBuf,
    backend: &Arc<CpuBackend>,
    mut model: MnistMLP<CpuBackend, f32>,
) -> Result<(), Box<dyn Error>> {
    type B = CpuBackend;
    type D = f32;

    let mut optimizer = Sgd::<B, D>::builder()
        .learning_rate(0.1)
        .init(&model.params())?;

    println!("Training for {} epochs ...", cfg.epochs);

    for epoch in 0..cfg.epochs {
        let rng = StdRng::seed_from_u64(cfg.seed + epoch as u64);
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
            });

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
                    "  epoch {:>2} | step {:>4} | step-loss = {:>8.4} | step-acc = {:>5.1}%",
                    epoch + 1,
                    step + 1,
                    loss_val,
                    acc * 100.0
                );
            }
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
        });

    let eval_ctx = Context::eval(backend);
    let mut total_correct = 0;
    let mut total_samples = 0;

    for batch_res in test_stream.iter() {
        let batch = batch_res?;
        let logits = model.forward(&eval_ctx, eval_ctx.input(&batch.images))?;

        let batch_size = batch.labels.shape()[0];
        let acc = accuracy_top1(&logits, &batch.labels)?;

        let batch_correct = (acc * batch_size as f32).round() as usize;
        total_correct += batch_correct;
        total_samples += batch_size;
    }

    let final_accuracy = if total_samples > 0 {
        (total_correct as f32 / total_samples as f32) * 100.0
    } else {
        0.0
    };

    println!(
        "\n  Final Test Accuracy: {:.2}% ({}/{} samples)",
        final_accuracy, total_correct, total_samples
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());
    let data_root = PathBuf::from(data_dir);

    println!("Using MNIST data directory: {:?}", data_root);
    mnist::ensure_downloaded(&data_root)?;

    let backend = Arc::new(CpuBackend::new());

    let model = MnistMLP::new(&backend, 128)?;

    println!("\n=== Custom MLP Model ===");
    run_experiment(
        &ExperimentConfig {
            seed: 42,
            batch_size: 128,
            epochs: 2,
            shuffle_buffer: 10_000,
        },
        &data_root,
        &backend,
        model,
    )?;

    Ok(())
}
