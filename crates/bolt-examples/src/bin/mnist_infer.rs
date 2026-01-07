#[path = "../mnist_common.rs"]
mod mnist_common;

use std::path::PathBuf;
use std::sync::Arc;

use bolt_datasets::mnist;
use bolt_serialize::{LoadOpts, load};

use mnist_common::{B, D, MnistMLP, evaluate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_root = std::env::var("MNIST_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| std::path::PathBuf::from("data/mnist"));
    mnist::ensure_downloaded(&data_root)?;

    let ckpt_dir: PathBuf = std::env::var("CKPT_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| PathBuf::from("data/mnist_ckpt"));

    let batch_size: usize = std::env::var("MNIST_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let backend = Arc::new(B::new());
    let mut store = bolt_nn::Store::<B, D>::new(backend.clone(), 1337);
    let model = MnistMLP::init(&store, 512, 256)?;

    store.group_params_by_name(|name| name.contains("bias"), 1);
    store.seal();

    let _ckpt_info = load(&mut store, &ckpt_dir, &LoadOpts::default())?;

    // Note: RNG state restoration would need to be added to bolt-serialize
    // For inference, RNG is not needed

    let eval = evaluate(&model, &data_root, backend, batch_size)?;
    println!(
        "Test Loss: {:.4}  Test Acc: {:.2}%",
        eval.loss,
        eval.acc * 100.0
    );
    println!(
        "Recovered test accuracy: {:.2}% ({}/{})",
        eval.acc * 100.0,
        eval.correct,
        eval.total
    );

    Ok(())
}
