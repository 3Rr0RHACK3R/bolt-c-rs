use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Linear;
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::{Tensor, no_grad};

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

fn set_param(param: &bolt_nn::Param<B, D>, backend: &Arc<B>, values: &[f32], shape: &[usize]) {
    let t = Tensor::<B, D>::from_slice(backend, values, shape).unwrap();
    param.set_tensor(t).unwrap();
}

#[test]
fn frozen_param_receives_no_gradient() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer = Linear::init(&store.sub("linear"), 2, 1, true).unwrap();
    set_param(&layer.weight, &backend, &[1.0, 2.0], &[1, 2]);
    set_param(layer.bias.as_ref().unwrap(), &backend, &[0.5], &[1]);

    layer.weight.freeze();

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = layer.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();

    store.backward(&loss).unwrap();

    assert!(layer.weight.grad().is_none());
    let b_grad = layer.bias.as_ref().unwrap().grad().expect("bias grad");
    assert_close(tensor_to_vec(&b_grad)[0], 1.0, 1e-5, "bias grad");
}

#[test]
fn unfreeze_restores_gradient_tracking() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer = Linear::init(&store.sub("linear"), 2, 1, false).unwrap();
    set_param(&layer.weight, &backend, &[1.0, 2.0], &[1, 2]);

    layer.weight.freeze();
    assert!(!layer.weight.requires_grad());
    layer.weight.unfreeze();
    assert!(layer.weight.requires_grad());

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = layer.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    let w_grad = layer.weight.grad().expect("weight grad");
    let vals = tensor_to_vec(&w_grad);
    assert_close(vals[0], 1.0, 1e-5, "w grad[0]");
    assert_close(vals[1], 2.0, 1e-5, "w grad[1]");
}

#[test]
fn shared_parameter_accumulates_gradients_across_multiple_uses() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let shared = store
        .param("shared", &[2, 2], bolt_nn::Init::Zeros)
        .unwrap();
    set_param(&shared, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    store.zero_grad();

    let w1 = shared.tensor();
    let w2 = shared.tensor();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();

    let y1 = x.matmul(&w1).unwrap();
    let y2 = y1.matmul(&w2.transpose(-1, -2).unwrap()).unwrap();
    let loss = y2.sum(None, false).unwrap();

    store.backward(&loss).unwrap();

    let grad = shared.grad().expect("shared grad");
    let vals = tensor_to_vec(&grad);
    assert_close(vals[0], 2.0, 1e-5, "grad[0,0]");
    assert_close(vals[1], 3.0, 1e-5, "grad[0,1]");
    assert_close(vals[2], 3.0, 1e-5, "grad[1,0]");
    assert_close(vals[3], 4.0, 1e-5, "grad[1,1]");
}

#[test]
fn inference_after_training_uses_updated_weights() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer = Linear::init(&store.sub("linear"), 2, 1, true).unwrap();
    set_param(&layer.weight, &backend, &[1.0, 2.0], &[1, 2]);
    set_param(layer.bias.as_ref().unwrap(), &backend, &[0.5], &[1]);

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = layer.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    let lr = 0.1f32;
    let w_grad = layer.weight.grad().unwrap().to_vec().unwrap();
    let mut w = layer.weight.tensor().to_vec().unwrap();
    for i in 0..w.len() {
        w[i] -= lr * w_grad[i];
    }

    let _ng = no_grad();
    layer
        .weight
        .set_tensor(Tensor::<B, D>::from_vec(&backend, w, &[1, 2]).unwrap())
        .unwrap();
    store.zero_grad();

    let updated = layer.weight.tensor().to_vec().unwrap();
    assert_close(updated[0], 0.9, 1e-5, "updated w[0]");
    assert_close(updated[1], 1.8, 1e-5, "updated w[1]");

    let x = Tensor::<B, D>::from_slice(&backend, &[3.0, 4.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let pred = layer.forward(x, &mut ctx).unwrap();
    let pred_val = tensor_to_vec(&pred)[0];
    assert_close(pred_val, 10.4, 1e-5, "inference output");
}

#[test]
fn low_level_tensor_backward_returns_expected_grads() {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let w = Tensor::<B, D>::from_slice(&backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .unwrap()
        .requires_grad();
    let b = Tensor::<B, D>::from_slice(&backend, &[0.1, 0.2], &[2])
        .unwrap()
        .requires_grad();

    let y = x.matmul(&w.transpose(-1, -2).unwrap()).unwrap();
    let b2 = b.broadcast_to(y.shape().as_slice()).unwrap();
    let y = y.add(&b2).unwrap();
    let loss = y.sum(None, false).unwrap();
    let grads = loss.backward().unwrap();

    let w_grad = grads.wrt(&w).unwrap().to_vec().unwrap();
    assert_close(w_grad[0], 1.0, 1e-5, "w grad[0,0]");
    assert_close(w_grad[1], 2.0, 1e-5, "w grad[0,1]");
    assert_close(w_grad[2], 1.0, 1e-5, "w grad[1,0]");
    assert_close(w_grad[3], 2.0, 1e-5, "w grad[1,1]");

    let b_grad = grads.wrt(&b).unwrap().to_vec().unwrap();
    assert_close(b_grad[0], 1.0, 1e-5, "b grad[0]");
    assert_close(b_grad[1], 1.0, 1e-5, "b grad[1]");
}

#[test]
fn manual_gradient_accumulation_across_batches_matches_expected_sum() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer = Linear::init(&store.sub("linear"), 2, 1, false).unwrap();
    set_param(&layer.weight, &backend, &[1.0, 1.0], &[1, 2]);

    let inputs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut acc = [0.0f32; 2];

    for x in inputs {
        store.zero_grad();
        let x = Tensor::<B, D>::from_slice(&backend, &x, &[1, 2]).unwrap();
        let mut ctx = ForwardCtx::eval();
        let y = layer.forward(x, &mut ctx).unwrap();
        let loss = y.sum(None, false).unwrap();
        store.backward(&loss).unwrap();

        let g = layer.weight.grad().unwrap().to_vec().unwrap();
        acc[0] += g[0];
        acc[1] += g[1];
    }

    assert_close(acc[0], 9.0, 1e-5, "acc grad[0]");
    assert_close(acc[1], 12.0, 1e-5, "acc grad[1]");
}

#[test]
fn multi_layer_gradient_flow_matches_expected_values() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer1 = Linear::init(&store.sub("l1"), 2, 2, false).unwrap();
    let layer2 = Linear::init(&store.sub("l2"), 2, 1, false).unwrap();

    set_param(&layer1.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    set_param(&layer2.weight, &backend, &[1.0, 1.0], &[1, 2]);

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let h = layer1.forward(x, &mut ctx).unwrap();
    let y = layer2.forward(h, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();

    store.backward(&loss).unwrap();

    let l2_grad = tensor_to_vec(&layer2.weight.grad().expect("l2 weight grad"));
    assert_close(l2_grad[0], 1.0, 1e-5, "l2 grad[0]");
    assert_close(l2_grad[1], 2.0, 1e-5, "l2 grad[1]");

    let l1_grad = tensor_to_vec(&layer1.weight.grad().expect("l1 weight grad"));
    assert_close(l1_grad[0], 1.0, 1e-5, "l1 grad[0,0]");
    assert_close(l1_grad[1], 2.0, 1e-5, "l1 grad[0,1]");
    assert_close(l1_grad[2], 1.0, 1e-5, "l1 grad[1,0]");
    assert_close(l1_grad[3], 2.0, 1e-5, "l1 grad[1,1]");
}

#[test]
fn alternating_optimization_with_freeze_unfreeze_isolated_grads_per_phase() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let model_g = Linear::init(&store.sub("g"), 2, 2, false).unwrap();
    let model_d = Linear::init(&store.sub("d"), 2, 1, false).unwrap();

    set_param(&model_g.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    set_param(&model_d.weight, &backend, &[1.0, 1.0], &[1, 2]);

    {
        model_g.weight.freeze();

        store.zero_grad();
        let noise = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let mut ctx = ForwardCtx::eval();
        let fake = model_g.forward(noise, &mut ctx).unwrap();
        let d_out = model_d.forward(fake, &mut ctx).unwrap();
        let d_loss = d_out.sum(None, false).unwrap();

        store.backward(&d_loss).unwrap();

        let d_grad = tensor_to_vec(&model_d.weight.grad().expect("d grad"));
        assert_close(d_grad[0], 1.0, 1e-5, "d grad[0] phase1");
        assert_close(d_grad[1], 2.0, 1e-5, "d grad[1] phase1");
        assert!(model_g.weight.grad().is_none());

        model_g.weight.unfreeze();
        store.zero_grad();
    }

    {
        model_d.weight.freeze();

        store.zero_grad();
        let noise = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
        let mut ctx = ForwardCtx::eval();
        let fake = model_g.forward(noise, &mut ctx).unwrap();
        let d_out = model_d.forward(fake, &mut ctx).unwrap();
        let g_loss = d_out.sum(None, false).unwrap();

        store.backward(&g_loss).unwrap();

        let g_grad = tensor_to_vec(&model_g.weight.grad().expect("g grad"));
        assert_close(g_grad[0], 1.0, 1e-5, "g grad[0,0] phase2");
        assert_close(g_grad[1], 2.0, 1e-5, "g grad[0,1] phase2");
        assert_close(g_grad[2], 1.0, 1e-5, "g grad[1,0] phase2");
        assert_close(g_grad[3], 2.0, 1e-5, "g grad[1,1] phase2");
        assert!(model_d.weight.grad().is_none());
    }
}

#[test]
fn shared_parameter_has_single_id() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let shared = store
        .param("shared", &[2, 2], bolt_nn::Init::Zeros)
        .unwrap();

    let p1 = shared.clone();
    let p2 = shared.clone();

    assert_eq!(p1.id(), p2.id());
    assert_eq!(p1.key(), p2.key());
}

#[test]
fn param_lookup_by_id_and_name() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let p = store
        .param("weight", &[2, 2], bolt_nn::Init::Zeros)
        .unwrap();
    let id = p.id();

    let by_id = store.param_by_id(id).unwrap();
    let by_name = store.param_by_name("weight").unwrap();

    assert_eq!(by_id.id(), id);
    assert_eq!(by_name.id(), id);
    assert_eq!(by_id.key(), "weight");
}

#[test]
fn buffer_lookup_by_id_and_name() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let b = store.buffer("buf", &[2, 2], bolt_nn::Init::Zeros).unwrap();
    let id = b.id();

    let by_id = store.buffer_by_id(id).unwrap();
    let by_name = store.buffer_by_name("buf").unwrap();

    assert_eq!(by_id.id(), id);
    assert_eq!(by_name.id(), id);
    assert_eq!(by_id.key(), "buf");
}

#[test]
fn name_to_id_and_id_to_name_mappings() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let _p1 = store.param("p1", &[2], bolt_nn::Init::Zeros).unwrap();
    let _p2 = store.param("p2", &[2], bolt_nn::Init::Zeros).unwrap();
    let _b1 = store.buffer("b1", &[2], bolt_nn::Init::Zeros).unwrap();

    let name_to_id = store.name_to_id();
    let id_to_name = store.id_to_name();

    assert_eq!(name_to_id.len(), 3);
    assert!(name_to_id.contains_key("p1"));
    assert!(name_to_id.contains_key("p2"));
    assert!(name_to_id.contains_key("b1"));

    assert_eq!(id_to_name.len(), 3);
    for (id, name) in &id_to_name {
        assert_eq!(name_to_id[name], *id);
    }
}

#[test]
fn each_param_gets_unique_id() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let p1 = store.param("p1", &[2], bolt_nn::Init::Zeros).unwrap();
    let p2 = store.param("p2", &[2], bolt_nn::Init::Zeros).unwrap();
    let b1 = store.buffer("b1", &[2], bolt_nn::Init::Zeros).unwrap();

    assert_ne!(p1.id(), p2.id());
    assert_ne!(p1.id(), b1.id());
    assert_ne!(p2.id(), b1.id());
}
