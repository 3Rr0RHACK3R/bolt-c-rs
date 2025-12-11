//! Edge case tests for the Unified Context API
//!
//! Tests based on the stress test document covering:
//! - Freeze/unfreeze (transfer learning)
//! - Parameter sharing (weight tying)
//! - Inference after training (zero overhead)
//! - Multiple forward passes in same context
//! - Gradient accumulation

use std::sync::Arc;

use bolt_autodiff::Parameter;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::linear;
use bolt_nn::{Context, Eval, Grad, Model};

type B = CpuBackend;
type D = f32;

// ============ Test 1: Freeze/Unfreeze (Transfer Learning) ============

#[test]
fn test_frozen_param_no_gradient() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    // Create layer and freeze weight
    let mut layer = linear(2, 1).build(&backend).unwrap();
    layer.weight.freeze();

    // Forward and backward
    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let loss = output.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

    // Frozen param should NOT have gradient
    assert!(
        layer.weight.grad().is_none(),
        "Frozen param should not receive gradient"
    );

    // Bias (not frozen) should have gradient
    assert!(
        layer.bias.as_ref().unwrap().grad().is_some(),
        "Unfrozen param should receive gradient"
    );
}

#[test]
fn test_unfreeze_enables_gradient() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    let mut layer = linear(2, 1).build(&backend).unwrap();

    layer.weight.freeze();
    assert!(!layer.weight.requires_grad());

    layer.weight.unfreeze();
    assert!(layer.weight.requires_grad());

    // Now should receive gradient
    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let loss = output.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

    assert!(
        layer.weight.grad().is_some(),
        "Unfrozen param should receive gradient"
    );
}

// ============ Test 2: param_frozen() Never Tracks ============

#[test]
fn test_param_frozen_never_tracks() {
    let backend = Arc::new(CpuBackend::new());

    let w_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[2, 1]).unwrap();
    let mut param = Parameter::new(w_data);

    // Also create a tracked param so backward has something to work with
    let other_data = Tensor::<B, D>::from_slice(&backend, &[0.5], &[1]).unwrap();
    let mut other_param = Parameter::new(other_data);

    // param has requires_grad = true by default
    assert!(param.requires_grad());

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let x_ad = grad_ctx.input(&x);
    let w_ad = grad_ctx.param_frozen(&param); // Force no tracking
    let other_ad = grad_ctx.param(&other_param); // This one is tracked

    let y = x_ad.matmul(&w_ad).unwrap().add(&other_ad).unwrap();
    let loss = y.sum(None, false).unwrap();

    grad_ctx
        .backward(&loss, &mut [&mut param, &mut other_param])
        .unwrap();

    // param_frozen should NOT have gradient even though we passed it to backward
    assert!(
        param.grad().is_none(),
        "param_frozen should never track gradients"
    );

    // The tracked param should have gradient
    assert!(
        other_param.grad().is_some(),
        "normally tracked param should have gradient"
    );
}

// ============ Test 3: Parameter Sharing (Weight Tying) ============

#[test]
fn test_parameter_sharing_deduplication() {
    let backend = Arc::new(CpuBackend::new());

    // Create one shared parameter
    let shared_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mut shared_param = Parameter::with_name(shared_data, "shared");

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let w1 = grad_ctx.param(&shared_param);
    let w2 = grad_ctx.param(&shared_param);

    let x = grad_ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());
    let y1 = x.matmul(&w1).unwrap();
    let y2 = y1.matmul(&w2.transpose(-1, -2).unwrap()).unwrap();
    let loss = y2.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut [&mut shared_param]).unwrap();

    // Should have accumulated gradient from both uses
    assert!(
        shared_param.grad().is_some(),
        "Shared param should receive gradient"
    );
}

// ============ Test 4: Inference After Training ============

#[test]
fn test_inference_after_training() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    let mut layer = linear(2, 1).build(&backend).unwrap();

    // Simulate training step
    {
        let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
        let loss = output.sum(None, false).unwrap();
        grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

        if let Some(grad) = layer.weight.grad() {
            let update = grad
                .mul(&Tensor::full(&backend, &[], 0.1f32).unwrap())
                .unwrap();
            *layer.weight.tensor_mut() = layer.weight.tensor().sub(&update).unwrap();
        }
        layer.weight.zero_grad();
    }

    // Inference - should work with Eval context (no autodiff overhead)
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);
    let test_x = Tensor::<B, D>::from_slice(&backend, &[3.0, 4.0], &[1, 2]).unwrap();
    let pred = layer.forward(&eval_ctx, eval_ctx.input(&test_x)).unwrap();

    // Output should be Tensor<B, D>, not Tensor<Autodiff<B, D>, D>
    assert_eq!(pred.shape(), &[1, 1]);
}

// ============ Test 5: Low-Level API (No bolt-nn layers) ============

#[test]
fn test_low_level_api_raw_params() {
    let backend = Arc::new(CpuBackend::new());

    // Raw parameters
    let w_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let b_data = Tensor::<B, D>::from_slice(&backend, &[0.1, 0.2], &[2]).unwrap();
    let mut w = Parameter::with_name(w_data, "weight");
    let mut b = Parameter::with_name(b_data, "bias");

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let x = grad_ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());
    let w_ad = grad_ctx.param(&w);
    let b_ad = grad_ctx.param(&b);

    // Manual forward: y = x @ W + b
    let y = x
        .matmul(&w_ad.transpose(-1, -2).unwrap())
        .unwrap()
        .add(&b_ad)
        .unwrap();
    let loss = y.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut [&mut w, &mut b]).unwrap();

    assert!(w.grad().is_some(), "Weight should have gradient");
    assert!(b.grad().is_some(), "Bias should have gradient");
}

// ============ Test 6: Multiple Layers Gradient Flow ============

#[test]
fn test_multi_layer_gradient_flow() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    let mut layer1 = linear(4, 3).build(&backend).unwrap();
    let mut layer2 = linear(3, 2).build(&backend).unwrap();

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

    // Forward through both layers
    let h = layer1.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let y = layer2.forward(&grad_ctx, h).unwrap();
    let loss = y.sum(None, false).unwrap();

    // Collect all params
    let mut all_params: Vec<&mut Parameter<B, D>> = layer1.params_mut();
    all_params.extend(layer2.params_mut());

    grad_ctx.backward(&loss, &mut all_params).unwrap();

    // All params should have gradients
    assert!(
        layer1.weight.grad().is_some(),
        "Layer1 weight should have gradient"
    );
    assert!(
        layer1.bias.as_ref().unwrap().grad().is_some(),
        "Layer1 bias should have gradient"
    );
    assert!(
        layer2.weight.grad().is_some(),
        "Layer2 weight should have gradient"
    );
    assert!(
        layer2.bias.as_ref().unwrap().grad().is_some(),
        "Layer2 bias should have gradient"
    );
}

// ============ Test 7: Separate Contexts for Train vs Predict ============

#[test]
fn test_gan_style_alternating_optimization() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    // Two models - simulate generator and discriminator
    let mut model_g = linear(2, 2).build(&backend).unwrap();
    let mut model_d = linear(2, 1).build(&backend).unwrap();

    // Phase 1: Train D with G frozen
    {
        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let noise = ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());

        // G forward with frozen params (no grad)
        let g_w = ctx.param_frozen(&model_g.weight);
        let g_b = ctx.param_frozen(model_g.bias.as_ref().unwrap());
        let fake = noise
            .matmul(&g_w.transpose(-1, -2).unwrap())
            .unwrap()
            .add(&g_b)
            .unwrap();

        // D forward with tracked params
        let d_out = model_d.forward(&ctx, fake).unwrap();
        let d_loss = d_out.sum(None, false).unwrap();

        ctx.backward(&d_loss, &mut model_d.params_mut()).unwrap();

        // D should have grad, G should not
        assert!(
            model_d.weight.grad().is_some(),
            "Discriminator should have grad"
        );
        assert!(
            model_g.weight.grad().is_none(),
            "Generator should not have grad (frozen)"
        );
    }

    // Phase 2: Train G with D frozen
    {
        model_d.weight.zero_grad();
        model_d.bias.as_mut().unwrap().zero_grad();

        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let noise = ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());

        // G forward with tracked params
        let fake = model_g.forward(&ctx, noise).unwrap();

        // D forward with frozen params
        let d_w = ctx.param_frozen(&model_d.weight);
        let d_b = ctx.param_frozen(model_d.bias.as_ref().unwrap());
        let d_out = fake
            .matmul(&d_w.transpose(-1, -2).unwrap())
            .unwrap()
            .add(&d_b)
            .unwrap();
        let g_loss = d_out.sum(None, false).unwrap();

        ctx.backward(&g_loss, &mut model_g.params_mut()).unwrap();

        assert!(
            model_g.weight.grad().is_some(),
            "Generator should have grad"
        );
        assert!(
            model_d.weight.grad().is_none(),
            "Discriminator should not have grad (frozen)"
        );
    }
}

// ============ Test 8: Eval Context Returns Base Tensor ============

#[test]
fn test_eval_context_returns_base_tensor() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    let layer = linear(2, 1).build(&backend).unwrap();

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&eval_ctx, eval_ctx.input(&x)).unwrap();

    // Output type is Tensor<B, D> = Tensor<CpuBackend, f32>, NOT Tensor<Autodiff<...>, f32>
    let _: Tensor<B, D> = output; // This compiles, proving it's the base type
}

// ============ Test 9: Fresh Context Per Batch ============

#[test]
fn test_fresh_context_per_batch() {
    let backend = Arc::new(CpuBackend::new());
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    let mut layer = linear(2, 1).build(&backend).unwrap();

    // Simulate 3 training batches
    for i in 0..3 {
        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

        let x_data = vec![i as f32 + 1.0, i as f32 + 2.0];
        let x = Tensor::<B, D>::from_slice(&backend, &x_data, &[1, 2]).unwrap();

        let output = layer.forward(&ctx, ctx.input(&x)).unwrap();
        let loss = output.sum(None, false).unwrap();

        ctx.backward(&loss, &mut layer.params_mut()).unwrap();

        // Check gradient exists
        assert!(
            layer.weight.grad().is_some(),
            "Batch {} should produce gradient",
            i
        );

        // Zero grad for next batch
        layer.weight.zero_grad();
        layer.bias.as_mut().unwrap().zero_grad();
    }
}
