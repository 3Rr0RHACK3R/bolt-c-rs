use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{Linear, Seq};
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::Tensor;

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
fn seq_gradients_flow_through_multiple_layers() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let l1 = Linear::init(&store.sub("l1"), 2, 2, false).unwrap();
    set_param(&l1.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let l2 = Linear::init(&store.sub("l2"), 2, 1, false).unwrap();
    set_param(&l2.weight, &backend, &[1.0, 1.0], &[1, 2]);

    let model: Seq<B, D> = Seq::new().push(l1).push(l2);

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = model.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    let mut l1_grad: Option<Vec<f32>> = None;
    let mut l2_grad: Option<Vec<f32>> = None;

    for (k, p) in store.named_trainable() {
        let Some(g) = p.grad() else { continue };
        let vals = tensor_to_vec(&g);
        match k.as_str() {
            "l1.weight" => l1_grad = Some(vals),
            "l2.weight" => l2_grad = Some(vals),
            _ => {}
        }
    }

    let l2_grad = l2_grad.expect("l2 grad missing");
    assert_close(l2_grad[0], 1.0, 1e-5, "l2 grad[0]");
    assert_close(l2_grad[1], 2.0, 1e-5, "l2 grad[1]");

    let l1_grad = l1_grad.expect("l1 grad missing");
    assert_close(l1_grad[0], 1.0, 1e-5, "l1 grad[0]");
    assert_close(l1_grad[1], 2.0, 1e-5, "l1 grad[1]");
    assert_close(l1_grad[2], 1.0, 1e-5, "l1 grad[2]");
    assert_close(l1_grad[3], 2.0, 1e-5, "l1 grad[3]");
}

#[test]
fn seq_freeze_unfreeze_and_zero_grad() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let layer = Linear::init(&store.sub("linear"), 1, 1, true).unwrap();
    set_param(&layer.weight, &backend, &[2.0], &[1, 1]);
    set_param(layer.bias.as_ref().unwrap(), &backend, &[0.5], &[1]);

    let model: Seq<B, D> = Seq::new().push(layer);

    for p in store.trainable() {
        p.freeze();
    }

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[3.0], &[1, 1]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = model.forward(x, &mut ctx).unwrap();
    let dummy = Tensor::<B, D>::from_slice(&backend, &[0.0], &[1, 1])
        .unwrap()
        .requires_grad();
    let y = y.add(&dummy).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    for p in store.trainable() {
        assert!(p.grad().is_none(), "frozen param should have no grad");
    }

    for p in store.trainable() {
        p.unfreeze();
    }

    store.zero_grad();
    let x = Tensor::<B, D>::from_slice(&backend, &[3.0], &[1, 1]).unwrap();
    let mut ctx = ForwardCtx::eval();
    let y = model.forward(x, &mut ctx).unwrap();
    let loss = y.sum(None, false).unwrap();
    store.backward(&loss).unwrap();

    let w_grad = store
        .named_trainable()
        .into_iter()
        .find(|(k, _)| k == "linear.weight")
        .and_then(|(_, p)| p.grad())
        .expect("weight grad missing");
    assert_close(tensor_to_vec(&w_grad)[0], 3.0, 1e-5, "weight grad");

    store.zero_grad();
    for p in store.trainable() {
        assert!(p.grad().is_none(), "zero_grad should clear gradients");
    }
}
