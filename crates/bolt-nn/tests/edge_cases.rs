//! Edge case tests for the Unified Context API
//!
//! Tests covering:
//! - Freeze/unfreeze (transfer learning)
//! - Parameter sharing (weight tying)
//! - Inference after training (zero overhead)
//! - Multiple forward passes in same context
//! - Gradient accumulation

use std::sync::Arc;

use bolt_autodiff::{AutodiffTensorExt, Parameter};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::{linear, HasParams};
use bolt_nn::{Context, Eval, Grad, Model};

type B = CpuBackend;
type D = f32;

fn assert_close(actual: f32, expected: f32, tol: f32, msg: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "{}: expected {}, got {}, diff {}",
        msg,
        expected,
        actual,
        diff
    );
}

fn tensor_to_vec(t: &Tensor<B, D>) -> Vec<f32> {
    t.to_vec().unwrap()
}

fn set_param(param: &mut Parameter<B, D>, backend: &Arc<B>, values: &[f32], shape: &[usize]) {
    *param.tensor_mut() = Tensor::<B, D>::from_slice(backend, values, shape).unwrap();
}

#[test]
fn test_frozen_param_no_gradient() {
    let backend = Arc::new(CpuBackend::new());

    // Create layer with known weights for deterministic gradient computation
    // y = x @ W^T + b, where W is [1, 2] and b is [0.5]
    // For x = [1, 2]: y = 1*1 + 2*2 + 0.5 = 5.5
    // loss = sum(y) = 5.5
    // dL/dW = x^T = [1, 2], dL/db = 1
    let mut layer = linear(2, 1).bias(true).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 2.0], &[1, 2]);
    set_param(layer.bias.as_mut().unwrap(), &backend, &[0.5], &[1]);

    // Freeze weight only
    layer.weight.freeze();

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let loss = output.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

    // Frozen weight should NOT have gradient
    assert!(
        layer.weight.grad().is_none(),
        "Frozen weight should not receive gradient"
    );

    // Bias (not frozen) should have gradient = 1.0
    let bias_grad = layer.bias.as_ref().unwrap().grad().expect("Bias should have gradient");
    let bias_grad_val = tensor_to_vec(bias_grad);
    assert_close(bias_grad_val[0], 1.0, 1e-5, "Bias gradient");
}

#[test]
fn test_unfreeze_restores_gradient_tracking() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 2.0], &[1, 2]);

    // Freeze then unfreeze
    layer.weight.freeze();
    assert!(!layer.weight.requires_grad());
    layer.weight.unfreeze();
    assert!(layer.weight.requires_grad());

    // For y = x @ W^T where x = [1, 2], W = [1, 2]
    // y = 1*1 + 2*2 = 5
    // loss = sum(y) = 5
    // dL/dW = x^T = [1, 2]
    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let loss = output.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

    let weight_grad = layer.weight.grad().expect("Unfrozen weight should have gradient");
    let grad_vals = tensor_to_vec(weight_grad);
    assert_close(grad_vals[0], 1.0, 1e-5, "Weight gradient[0]");
    assert_close(grad_vals[1], 2.0, 1e-5, "Weight gradient[1]");
}

#[test]
fn test_param_frozen_overrides_requires_grad_flag() {
    // param_frozen() should NOT track even when requires_grad=true
    let backend = Arc::new(CpuBackend::new());

    // param_a stays frozen even though requires_grad is true; param_b remains trainable
    let w_data = Tensor::<B, D>::from_slice(&backend, &[2.0, 3.0], &[2, 1]).unwrap();
    let mut param_a = Parameter::new(w_data);
    let other_data = Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap();
    let mut param_b = Parameter::new(other_data);

    assert!(param_a.requires_grad(), "param_a should require grad by default");

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    // x @ param_a + param_b where x=[1,2]; only param_b should pick up grad=1
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let x_ad = grad_ctx.input(&x);
    let a_ad = grad_ctx.param_frozen(&param_a); // Force frozen despite requires_grad=true
    let b_ad = grad_ctx.param(&param_b);

    let y = x_ad.matmul(&a_ad).unwrap().add(&b_ad).unwrap();
    let loss = y.sum(None, false).unwrap();

    grad_ctx
        .backward(&loss, &mut [&mut param_a, &mut param_b])
        .unwrap();

    // param_frozen should NOT have gradient
    assert!(
        param_a.grad().is_none(),
        "param_frozen should never track gradients, even with requires_grad=true"
    );

    // param_b tracked normally
    let b_grad = param_b.grad().expect("param_b should have gradient");
    assert_close(tensor_to_vec(b_grad)[0], 1.0, 1e-5, "param_b gradient");
}

#[test]
fn test_parameter_sharing_accumulates_gradients() {
    // Shared weight is used twice; gradients from both uses should add up.
    // For identity W and x=[1,2], total dL/dW should be [[2,3],[3,4]].
    let backend = Arc::new(CpuBackend::new());

    // W = [[1,0],[0,1]] and x=[[1,2]]
    let shared_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let mut shared_param = Parameter::with_name(shared_data, "shared");

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let w1 = grad_ctx.param(&shared_param);
    let w2 = grad_ctx.param(&shared_param);

    let x = grad_ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());
    let y1 = x.matmul(&w1).unwrap();
    let y2 = y1.matmul(&w2.transpose(-1, -2).unwrap()).unwrap();
    let loss = y2.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut [&mut shared_param]).unwrap();

    let grad = shared_param.grad().expect("Shared param should have gradient");
    let grad_vals = tensor_to_vec(grad);

    // Expected accumulated gradient: [[2, 3], [3, 4]] (row-major)
    assert_close(grad_vals[0], 2.0, 1e-5, "grad[0,0]");
    assert_close(grad_vals[1], 3.0, 1e-5, "grad[0,1]");
    assert_close(grad_vals[2], 3.0, 1e-5, "grad[1,0]");
    assert_close(grad_vals[3], 4.0, 1e-5, "grad[1,1]");
}

#[test]
fn test_inference_after_training_produces_correct_output() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 1).bias(true).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 2.0], &[1, 2]);
    set_param(layer.bias.as_mut().unwrap(), &backend, &[0.5], &[1]);

    // Training step: update weights
    {
        let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let output = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
        let loss = output.sum(None, false).unwrap();
        grad_ctx.backward(&loss, &mut layer.params_mut()).unwrap();

        // Simple SGD update: W -= lr * grad
        // dL/dW = [1, 2], lr = 0.1
        // W_new = [1, 2] - 0.1 * [1, 2] = [0.9, 1.8]
        if let Some(grad) = layer.weight.grad() {
            let lr = Tensor::full(&backend, &[], 0.1f32).unwrap();
            let update = grad.mul(&lr).unwrap();
            *layer.weight.tensor_mut() = layer.weight.tensor().sub(&update).unwrap();
        }
        layer.zero_grad(); // Uses HasParams trait method
    }

    // Verify weight was updated
    let w_vals = tensor_to_vec(layer.weight.tensor());
    assert_close(w_vals[0], 0.9, 1e-5, "Updated weight[0]");
    assert_close(w_vals[1], 1.8, 1e-5, "Updated weight[1]");

    // Inference with Eval context
    // y = x @ W^T + b = [3, 4] @ [0.9, 1.8]^T + 0.5 = 3*0.9 + 4*1.8 + 0.5 = 2.7 + 7.2 + 0.5 = 10.4
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);
    let test_x = Tensor::<B, D>::from_slice(&backend, &[3.0, 4.0], &[1, 2]).unwrap();
    let pred = layer.forward(&eval_ctx, eval_ctx.input(&test_x)).unwrap();

    let pred_vals = tensor_to_vec(&pred);
    assert_close(pred_vals[0], 10.4, 1e-5, "Inference output");
}

#[test]
fn test_low_level_api_gradient_values() {
    let backend = Arc::new(CpuBackend::new());

    // Simple linear: x=[[1,2]], W=I, b=[0.1,0.2]; loss=sum(y)=3.3.
    // Expected grads: dW=[[1,2],[1,2]], db=[1,1].
    let w_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let b_data = Tensor::<B, D>::from_slice(&backend, &[0.1, 0.2], &[2]).unwrap();
    let mut w = Parameter::with_name(w_data, "weight");
    let mut b = Parameter::with_name(b_data, "bias");

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

    let x = grad_ctx.input(&Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap());
    let w_ad = grad_ctx.param(&w);
    let b_ad = grad_ctx.param(&b);

    let y = x.matmul(&w_ad.transpose(-1, -2).unwrap()).unwrap().add(&b_ad).unwrap();
    let loss = y.sum(None, false).unwrap();

    grad_ctx.backward(&loss, &mut [&mut w, &mut b]).unwrap();

    let w_grad = tensor_to_vec(w.grad().expect("Weight gradient"));
    // dL/dW = [[1, 2], [1, 2]] (row-major: [1, 2, 1, 2])
    assert_close(w_grad[0], 1.0, 1e-5, "W grad[0,0]");
    assert_close(w_grad[1], 2.0, 1e-5, "W grad[0,1]");
    assert_close(w_grad[2], 1.0, 1e-5, "W grad[1,0]");
    assert_close(w_grad[3], 2.0, 1e-5, "W grad[1,1]");

    let b_grad = tensor_to_vec(b.grad().expect("Bias gradient"));
    assert_close(b_grad[0], 1.0, 1e-5, "b grad[0]");
    assert_close(b_grad[1], 1.0, 1e-5, "b grad[1]");
}

#[test]
fn test_multi_layer_gradient_flow_with_values() {
    let backend = Arc::new(CpuBackend::new());

    // Two linear layers with W1=I and W2=[1,1]; x=[1,2] so loss=3.
    // Expected grads: dW2=[1,2], dW1=[[1,2],[1,2]].
    let mut layer1 = linear(2, 2).bias(false).build(&backend).unwrap();
    set_param(&mut layer1.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let mut layer2 = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut layer2.weight, &backend, &[1.0, 1.0], &[1, 2]);

    let grad_ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();

    let h = layer1.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let y = layer2.forward(&grad_ctx, h).unwrap();
    let loss = y.sum(None, false).unwrap();

    let mut all_params: Vec<&mut Parameter<B, D>> = layer1.params_mut();
    all_params.extend(layer2.params_mut());
    grad_ctx.backward(&loss, &mut all_params).unwrap();

    // Verify L2 gradient: [[1, 2]]
    let l2_grad = tensor_to_vec(layer2.weight.grad().expect("L2 weight gradient"));
    assert_close(l2_grad[0], 1.0, 1e-5, "L2 grad[0]");
    assert_close(l2_grad[1], 2.0, 1e-5, "L2 grad[1]");

    // Verify L1 gradient: [[1, 2], [1, 2]]
    let l1_grad = tensor_to_vec(layer1.weight.grad().expect("L1 weight gradient"));
    assert_close(l1_grad[0], 1.0, 1e-5, "L1 grad[0,0]");
    assert_close(l1_grad[1], 2.0, 1e-5, "L1 grad[0,1]");
    assert_close(l1_grad[2], 1.0, 1e-5, "L1 grad[1,0]");
    assert_close(l1_grad[3], 2.0, 1e-5, "L1 grad[1,1]");
}

#[test]
fn test_alternating_optimization_with_freeze_unfreeze() {
    // Freeze/unfreeze across phases: train D, then train G, using the same path.
    let backend = Arc::new(CpuBackend::new());

    // G: identity; D: weights [1,1]
    let mut model_g = linear(2, 2).bias(false).build(&backend).unwrap();
    set_param(&mut model_g.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let mut model_d = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut model_d.weight, &backend, &[1.0, 1.0], &[1, 2]);

    // Phase 1: train D only; G is frozen so it should stay gradient-free
    {
        model_g.freeze();

        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let noise = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let fake = model_g.forward(&ctx, ctx.input(&noise)).unwrap();
        let d_out = model_d.forward(&ctx, fake).unwrap();
        let d_loss = d_out.sum(None, false).unwrap();

        ctx.backward(&d_loss, &mut model_d.params_mut()).unwrap();

        // D gradient: [[1, 2]]
        let d_grad = model_d.weight.grad().expect("D should have gradient");
        let d_grad_vals = tensor_to_vec(d_grad);
        assert_close(d_grad_vals[0], 1.0, 1e-5, "D grad[0] in phase 1");
        assert_close(d_grad_vals[1], 2.0, 1e-5, "D grad[1] in phase 1");

        // G should have no gradient (frozen)
        assert!(
            model_g.weight.grad().is_none(),
            "G should have no gradient when frozen"
        );

        model_g.unfreeze();
        model_d.zero_grad();
    }

    // Phase 2: train G only; D stays frozen. G grad should be [[1,2],[1,2]].
    {
        model_d.freeze();

        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let noise = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let fake = model_g.forward(&ctx, ctx.input(&noise)).unwrap();
        let d_out = model_d.forward(&ctx, fake).unwrap();
        let g_loss = d_out.sum(None, false).unwrap();

        ctx.backward(&g_loss, &mut model_g.params_mut()).unwrap();

        // G gradient: [[1, 2], [1, 2]]
        let g_grad = model_g.weight.grad().expect("G should have gradient");
        let g_grad_vals = tensor_to_vec(g_grad);
        assert_close(g_grad_vals[0], 1.0, 1e-5, "G grad[0,0] in phase 2");
        assert_close(g_grad_vals[1], 2.0, 1e-5, "G grad[0,1] in phase 2");
        assert_close(g_grad_vals[2], 1.0, 1e-5, "G grad[1,0] in phase 2");
        assert_close(g_grad_vals[3], 2.0, 1e-5, "G grad[1,1] in phase 2");

        // D should have no gradient (frozen)
        assert!(
            model_d.weight.grad().is_none(),
            "D should have no gradient when frozen"
        );
    }
}

#[test]
fn test_eval_context_produces_correct_values() {
    let backend = Arc::new(CpuBackend::new());

    let layer = {
        let mut l = linear(2, 1).bias(true).build(&backend).unwrap();
        set_param(&mut l.weight, &backend, &[2.0, 3.0], &[1, 2]);
        set_param(l.bias.as_mut().unwrap(), &backend, &[0.5], &[1]);
        l
    };

    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);

    // y = x @ W^T + b = [1, 2] @ [2, 3]^T + 0.5 = 2 + 6 + 0.5 = 8.5
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&eval_ctx, eval_ctx.input(&x)).unwrap();

    let out_vals = tensor_to_vec(&output);
    assert_close(out_vals[0], 8.5, 1e-5, "Eval output");
}

#[test]
fn test_eval_and_grad_produce_same_forward_values() {
    let backend = Arc::new(CpuBackend::new());

    let layer = {
        let mut l = linear(3, 2).bias(true).build(&backend).unwrap();
        set_param(
            &mut l.weight,
            &backend,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        );
        set_param(l.bias.as_mut().unwrap(), &backend, &[0.1, 0.2], &[2]);
        l
    };

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0], &[1, 3]).unwrap();

    // Eval context
    let eval_ctx = Context::eval(&backend);
    let eval_out = layer.forward(&eval_ctx, eval_ctx.input(&x)).unwrap();
    let eval_vals = eval_out.to_vec().unwrap();

    // Grad context
    let grad_ctx = Context::grad(&backend);
    let grad_out = layer.forward(&grad_ctx, grad_ctx.input(&x)).unwrap();
    let grad_vals = grad_out.to_vec().unwrap();

    assert_close(eval_vals[0], grad_vals[0], 1e-5, "Output[0] should match");
    assert_close(eval_vals[1], grad_vals[1], 1e-5, "Output[1] should match");
}

#[test]
fn test_fresh_context_per_batch_produces_independent_gradients() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 1.0], &[1, 2]);

    // Each batch should produce gradient = input (since dL/dW = x^T for sum loss)
    let inputs = vec![
        vec![1.0, 2.0], // grad should be [1, 2]
        vec![3.0, 4.0], // grad should be [3, 4]
        vec![5.0, 6.0], // grad should be [5, 6]
    ];

    for (i, input) in inputs.iter().enumerate() {
        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let x = Tensor::<B, D>::from_slice(&backend, input, &[1, 2]).unwrap();

        let output = layer.forward(&ctx, ctx.input(&x)).unwrap();
        let loss = output.sum(None, false).unwrap();

        ctx.backward(&loss, &mut layer.params_mut()).unwrap();

        let grad = layer.weight.grad().expect("Should have gradient");
        let grad_vals = tensor_to_vec(grad);

        assert_close(
            grad_vals[0],
            input[0],
            1e-5,
            &format!("Batch {} grad[0]", i),
        );
        assert_close(
            grad_vals[1],
            input[1],
            1e-5,
            &format!("Batch {} grad[1]", i),
        );

        layer.zero_grad(); // Uses HasParams trait method
    }
}

#[test]
fn test_has_params_freeze_unfreeze_all() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 2).bias(true).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    set_param(layer.bias.as_mut().unwrap(), &backend, &[0.1, 0.2], &[2]);

    // Verify all params require grad by default
    assert!(layer.weight.requires_grad());
    assert!(layer.bias.as_ref().unwrap().requires_grad());

    // Freeze all via trait method
    layer.freeze();
    assert!(!layer.weight.requires_grad());
    assert!(!layer.bias.as_ref().unwrap().requires_grad());

    // Forward + backward should produce no gradients for frozen layer
    // Need at least one tracked tensor for backward to work
    let ctx = Context::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&ctx, ctx.input(&x)).unwrap();

    // Add a tracked dummy param that matches output shape for the add
    let ad_backend = ctx.autodiff();
    let dummy_ad = Tensor::from_slice(&ad_backend, &[1.0, 1.0], &[1, 2]).unwrap().requires_grad();
    let output_with_dummy = output.add(&dummy_ad).unwrap();
    let loss = output_with_dummy.sum(None, false).unwrap();

    let grads = loss.backward().unwrap();

    assert!(layer.weight.grad().is_none(), "Frozen weight should have no grad");
    assert!(
        layer.bias.as_ref().unwrap().grad().is_none(),
        "Frozen bias should have no grad"
    );
    assert!(grads.wrt(&dummy_ad).is_some(), "Dummy param should have grad");

    // Unfreeze and verify gradients flow
    layer.unfreeze();
    assert!(layer.weight.requires_grad());
    assert!(layer.bias.as_ref().unwrap().requires_grad());

    let ctx2 = Context::grad(&backend);
    let output2 = layer.forward(&ctx2, ctx2.input(&x)).unwrap();
    let loss2 = output2.sum(None, false).unwrap();
    ctx2.backward(&loss2, &mut layer.params_mut()).unwrap();

    assert!(layer.weight.grad().is_some(), "Unfrozen weight should have grad");
    assert!(
        layer.bias.as_ref().unwrap().grad().is_some(),
        "Unfrozen bias should have grad"
    );
}

#[test]
fn test_has_params_zero_grad_all() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 1).bias(true).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 2.0], &[1, 2]);
    set_param(layer.bias.as_mut().unwrap(), &backend, &[0.5], &[1]);

    // Generate gradients
    let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let output = layer.forward(&ctx, ctx.input(&x)).unwrap();
    let loss = output.sum(None, false).unwrap();
    ctx.backward(&loss, &mut layer.params_mut()).unwrap();

    assert!(layer.weight.grad().is_some());
    assert!(layer.bias.as_ref().unwrap().grad().is_some());

    // Zero all via trait method
    layer.zero_grad();

    assert!(layer.weight.grad().is_none(), "Weight grad should be cleared");
    assert!(
        layer.bias.as_ref().unwrap().grad().is_none(),
        "Bias grad should be cleared"
    );
}

#[test]
fn test_manual_gradient_accumulation_across_batches() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[1.0, 1.0], &[1, 2]);

    // Accumulate grads across three batches before updating; expected sum is [9, 12].
    let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

    let mut accumulated_grad: Option<Tensor<B, D>> = None;

    for input in &inputs {
        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
        let x = Tensor::<B, D>::from_slice(&backend, input, &[1, 2]).unwrap();

        let output = layer.forward(&ctx, ctx.input(&x)).unwrap();
        let loss = output.sum(None, false).unwrap();

        ctx.backward(&loss, &mut layer.params_mut()).unwrap();

        // Manually accumulate
        if let Some(grad) = layer.weight.grad() {
            accumulated_grad = Some(match accumulated_grad {
                Some(acc) => acc.add(grad).unwrap(),
                None => grad.clone(),
            });
        }

        layer.zero_grad(); // Uses HasParams trait method
    }

    let acc = accumulated_grad.expect("Should have accumulated gradient");
    let acc_vals = tensor_to_vec(&acc);
    assert_close(acc_vals[0], 9.0, 1e-5, "Accumulated grad[0]");
    assert_close(acc_vals[1], 12.0, 1e-5, "Accumulated grad[1]");
}
