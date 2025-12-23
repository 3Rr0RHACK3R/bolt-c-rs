use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Dropout;
use bolt_nn::{ForwardCtx, Module};
use bolt_rng::RngStreams;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn dropout_eval_is_identity() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = dropout.forward(x.clone(), &mut ctx).unwrap();
    assert_eq!(y.to_vec().unwrap(), x.to_vec().unwrap());
}

#[test]
fn dropout_same_seed_same_output() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();

    let mut ctx1 = ForwardCtx::train_with_rngs(RngStreams::from_seed(123));
    let y1 = dropout.forward(x.clone(), &mut ctx1).unwrap();

    let mut ctx2 = ForwardCtx::train_with_rngs(RngStreams::from_seed(123));
    let y2 = dropout.forward(x, &mut ctx2).unwrap();

    assert_eq!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

#[test]
fn dropout_advances_rng_between_calls() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let mut ctx = ForwardCtx::train_with_rngs(RngStreams::from_seed(999));

    let y1 = dropout.forward(x.clone(), &mut ctx).unwrap();
    let y2 = dropout.forward(x, &mut ctx).unwrap();

    assert_ne!(y1.to_vec().unwrap(), y2.to_vec().unwrap());
}

#[test]
fn dropout_train_ctx_has_default_rngs() {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let dropout = Dropout::new(0.5).unwrap();
    let mut ctx = ForwardCtx::train();
    dropout.forward(x, &mut ctx).unwrap();
}
