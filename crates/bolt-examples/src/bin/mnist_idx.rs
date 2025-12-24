use std::path::Path;
use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_data::Stream;
use bolt_datasets::mnist::{self, INPUT_DIM, MnistBatch, NUM_CLASSES};
use bolt_losses::{Reduction, accuracy_top1, cross_entropy_from_logits_sparse};
use bolt_nn::layers::Linear;
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_optim::{Sgd, SgdCfg, SgdGroupCfg};
use bolt_rng::RngStream;
use bolt_tensor::{Tensor, no_grad};
use bolt_vision::{ops::tensor, types::ImageLayout};

type B = CpuBackend;
type D = f32;
type Batch = MnistBatch<B>;
type BoxErr = Box<dyn std::error::Error>;

const MEAN: [f32; 1] = [0.1307];
const STD: [f32; 1] = [0.3081];

struct MnistMLP {
    fc1: Linear<B, D>,
    fc2: Linear<B, D>,
}

impl MnistMLP {
    fn init(store: &Store<B, D>, hidden: usize) -> bolt_nn::Result<Self> {
        let fc1 = Linear::init(&store.sub("fc1"), INPUT_DIM, hidden, true)?;
        let fc2 = Linear::init(&store.sub("fc2"), hidden, NUM_CLASSES, true)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module<B, D> for MnistMLP {
    fn forward(&self, x: Tensor<B, D>, ctx: &mut ForwardCtx) -> bolt_nn::Result<Tensor<B, D>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(bolt_nn::Error::Shape(format!(
                "expected [batch, channels, height, width], got {:?}",
                shape
            )));
        }
        let batch = shape[0];
        let x = x.reshape(&[batch, INPUT_DIM])?;
        let x = self.fc1.forward(x, ctx)?;
        let x = x.relu()?;
        self.fc2.forward(x, ctx)
    }
}

fn train_loader(
    root: &Path,
    backend: Arc<B>,
    batch_size: usize,
    rng: RngStream,
) -> Result<Stream<Batch>, BoxErr> {
    let stream = mnist::train(root)?
        .map_with(backend.clone(), |b, ex| mnist::to_tensor_label(b, ex))
        .try_map(|mut s| {
            s.image = tensor::scale(1.0 / 255.0, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .try_map(|mut s| {
            s.image = tensor::normalize(&MEAN, &STD, ImageLayout::NCHW, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .shuffle(10_000, rng)
        .batch(batch_size)
        .try_map_with(backend, |b, xs| {
            let images = Tensor::stack(b, xs.iter().map(|s| &s.image))?;
            let labels = Tensor::from_iter(b, xs.iter().map(|s| s.label))?;
            Ok::<_, bolt_data::DataError>(MnistBatch { images, labels })
        });
        // .take(10);
    Ok(stream)
}

fn test_loader(root: &Path, backend: Arc<B>, batch_size: usize) -> Result<Stream<Batch>, BoxErr> {
    let stream = mnist::test(root)?
        .map_with(backend.clone(), |b, ex| mnist::to_tensor_label(b, ex))
        .try_map(|mut s| {
            s.image = tensor::scale(1.0 / 255.0, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .try_map(|mut s| {
            s.image = tensor::normalize(&MEAN, &STD, ImageLayout::NCHW, s.image)?;
            Ok::<_, bolt_data::DataError>(s)
        })
        .batch(batch_size)
        .try_map_with(backend, |b, xs| {
            let images = Tensor::stack(b, xs.iter().map(|s| &s.image))?;
            let labels = Tensor::from_iter(b, xs.iter().map(|s| s.label))?;
            Ok::<_, bolt_data::DataError>(MnistBatch { images, labels })
        })
        .take(10);
    Ok(stream)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_root = std::env::var("MNIST_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| std::path::PathBuf::from("data/mnist"));
    mnist::ensure_downloaded(&data_root)?;

    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);
    let model = MnistMLP::init(&store, 128)?;

    store.group_params_by_name(|name| name.contains("bias"), 1);
    store.seal();

    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 0.1,
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

    let seed = 42u64;
    let batch_size = 128;
    let epochs = 1;

    for epoch in 0..epochs {
        let rng = RngStream::from_seed(seed + epoch as u64);
        let loader = train_loader(&data_root, backend.clone(), batch_size, rng)?;

        for (step, batch_res) in loader.iter().enumerate() {
            let batch = batch_res?;
            store.zero_grad();

            let mut ctx = ForwardCtx::train();
            let logits = model.forward(batch.images, &mut ctx)?;
            let loss = cross_entropy_from_logits_sparse(
                &logits,
                &batch.labels,
                NUM_CLASSES,
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
    }

    evaluate(&model, &data_root, backend, batch_size)?;
    Ok(())
}

fn evaluate(
    model: &MnistMLP,
    data_root: &Path,
    backend: Arc<B>,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Evaluation on Test Set ===");
    let _guard = no_grad();

    let loader = test_loader(data_root, backend, batch_size)?;

    let mut total_loss = 0.0f64;
    let mut total_correct = 0usize;
    let mut total_samples = 0usize;

    for batch_res in loader.iter() {
        let batch = batch_res?;
        let n = batch.labels.numel();

        let mut ctx = ForwardCtx::eval();
        let logits = model.forward(batch.images, &mut ctx)?;
        let loss =
            cross_entropy_from_logits_sparse(&logits, &batch.labels, NUM_CLASSES, Reduction::Mean)?;

        let loss_val: f64 = loss.item()?.into();
        total_loss += loss_val * n as f64;

        let preds = logits.argmax(Some(&[-1]), false)?;
        let correct = preds
            .to_vec()?
            .iter()
            .zip(batch.labels.to_vec()?.iter())
            .filter(|(p, t)| p == t)
            .count();
        total_correct += correct;
        total_samples += n;
    }

    let test_loss = total_loss / total_samples as f64;
    let test_acc = total_correct as f64 / total_samples as f64;
    println!(
        "Test Loss: {:.4}  Test Acc: {:.2}%",
        test_loss,
        test_acc * 100.0
    );

    Ok(())
}
