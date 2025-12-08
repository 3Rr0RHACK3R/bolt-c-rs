use std::sync::Arc;

use bolt_core::Backend;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::{linear, relu, ModelExt, Seq};
use bolt_nn::{Context, Model, Mode};

type B = CpuBackend;
type D = f32;

#[test]
fn test_linear_forward() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B>::infer(backend.clone());
    
    let spec = linear(4, 2);
    let layer = spec.build(&ctx).unwrap();
    
    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let output = layer.forward(&ctx, input).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_relu_forward() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B>::infer(backend.clone());
    
    let input = Tensor::<B, D>::from_slice(&backend, &[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
    let layer = relu();
    let output = layer.forward(&ctx, input).unwrap();
    
    let data = output.to_vec().unwrap();
    assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_seq_container() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B>::infer(backend.clone());
    
    let l1 = linear(4, 3).build(&ctx).unwrap();
    let l2 = linear(3, 2).build(&ctx).unwrap();
    
    let model: Seq<B, D> = Seq::new()
        .push(l1)
        .push(relu())
        .push(l2);
    
    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let output = model.forward(&ctx, input).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_then_combinator() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B>::infer(backend.clone());
    
    let l1 = linear(4, 3).build(&ctx).unwrap();
    let l2 = linear(3, 2).build(&ctx).unwrap();
    
    let model = l1.then(relu()).then(l2);
    
    let input = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let output = model.forward(&ctx, input).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_context_mode_switching() {
    let backend = Arc::new(CpuBackend::new());
    
    // Start in training mode
    let ctx = Context::<B>::train(backend.clone(), 42);
    assert!(ctx.is_training());
    assert_eq!(ctx.mode(), Mode::Train);
    
    // Switch to eval
    ctx.set_eval();
    assert!(!ctx.is_training());
    assert_eq!(ctx.mode(), Mode::Eval);
    
    // Switch back to train
    ctx.set_train();
    assert!(ctx.is_training());
    
    // Test scoped evaluating
    ctx.evaluating(|| {
        assert!(!ctx.is_training());
    });
    // Mode restored after scope
    assert!(ctx.is_training());
}

#[test]
fn test_context_backend_access() {
    let backend = Arc::new(CpuBackend::new());
    let ctx = Context::<B>::infer(backend.clone());
    
    // Can access backend through context
    let _ = ctx.backend();
    let _ = ctx.device();
}
