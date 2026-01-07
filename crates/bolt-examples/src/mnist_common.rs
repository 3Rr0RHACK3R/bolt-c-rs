use std::path::Path;
use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_data::Stream;
use bolt_datasets::mnist::{self, INPUT_DIM, MnistBatch, NUM_CLASSES};
use bolt_losses::{Reduction, cross_entropy_from_logits_sparse};
use bolt_nn::layers::{BatchNorm, Linear};
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::{Tensor, no_grad};
use bolt_vision::{ops::tensor, types::ImageLayout};

pub type B = CpuBackend;
pub type D = f32;
pub type Batch = MnistBatch<B>;
pub type BoxErr = Box<dyn std::error::Error>;

const MEAN: [f32; 1] = [0.1307];
const STD: [f32; 1] = [0.3081];

pub struct MnistMLP {
    fc1: Linear<B, D>,
    bn1: BatchNorm<B, D>,
    fc2: Linear<B, D>,
    bn2: BatchNorm<B, D>,
    fc3: Linear<B, D>,
}

impl MnistMLP {
    pub fn init(store: &Store<B, D>, hidden1: usize, hidden2: usize) -> bolt_nn::Result<Self> {
        let fc1 = Linear::init(&store.sub("fc1"), INPUT_DIM, hidden1, true)?;
        let bn1 = BatchNorm::init_default(&store.sub("bn1"), hidden1)?;
        let fc2 = Linear::init(&store.sub("fc2"), hidden1, hidden2, true)?;
        let bn2 = BatchNorm::init_default(&store.sub("bn2"), hidden2)?;
        let fc3 = Linear::init(&store.sub("fc3"), hidden2, NUM_CLASSES, true)?;
        Ok(Self {
            fc1,
            bn1,
            fc2,
            bn2,
            fc3,
        })
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
        let x = self.bn1.forward(x, ctx)?;
        let x = x.relu()?;
        let x = self.fc2.forward(x, ctx)?;
        let x = self.bn2.forward(x, ctx)?;
        let x = x.relu()?;
        self.fc3.forward(x, ctx)
    }
}

#[allow(dead_code)]
pub fn train_loader(
    root: &Path,
    backend: Arc<B>,
    batch_size: usize,
    key: bolt_rng::RngKey,
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
        .shuffle(10_000, key)
        .batch(batch_size)
        .try_map_with(backend, |b, xs| {
            let images = Tensor::stack(b, xs.iter().map(|s| &s.image))?;
            let labels = Tensor::from_iter(b, xs.iter().map(|s| s.label))?;
            Ok::<_, bolt_data::DataError>(MnistBatch { images, labels })
        });
    Ok(stream)
}

pub fn test_loader(
    root: &Path,
    backend: Arc<B>,
    batch_size: usize,
) -> Result<Stream<Batch>, BoxErr> {
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
        });
    Ok(stream)
}

#[derive(Clone, Copy, Debug)]
pub struct EvalResult {
    pub loss: f64,
    pub acc: f64,
    pub correct: usize,
    pub total: usize,
}

pub fn evaluate(
    model: &MnistMLP,
    data_root: &Path,
    backend: Arc<B>,
    batch_size: usize,
) -> Result<EvalResult, BoxErr> {
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
    Ok(EvalResult {
        loss: test_loss,
        acc: test_acc,
        correct: total_correct,
        total: total_samples,
    })
}
