use std::sync::Arc;

use bolt_autodiff::Parameter;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::{HasParams, Seq, linear};
use bolt_nn::{Context, Grad, Model};

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
fn test_seq_params_and_grad_flow() {
    let backend = Arc::new(CpuBackend::new());

    let mut l1 = linear(2, 2).bias(false).build(&backend).unwrap();
    set_param(&mut l1.weight, &backend, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let mut l2 = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut l2.weight, &backend, &[1.0, 1.0], &[1, 2]);

    let mut model: Seq<B, D> = Seq::new().push(l1).push(l2);

    let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0], &[1, 2]).unwrap();

    let y = model.forward(&ctx, ctx.input(&x)).unwrap();
    let loss = y.sum(None, false).unwrap();

    ctx.backward(&loss, &mut model.params_mut()).unwrap();

    let params = model.params();
    assert_eq!(params.len(), 2);

    let mut l1_grad: Option<Vec<f32>> = None;
    let mut l2_grad: Option<Vec<f32>> = None;

    for p in params {
        if let Some(g) = p.grad() {
            let vals = tensor_to_vec(g);
            match vals.len() {
                4 => l1_grad = Some(vals),
                2 => l2_grad = Some(vals),
                _ => panic!("unexpected gradient length {}", vals.len()),
            }
        }
    }

    let l1_grad = l1_grad.expect("layer1 grad missing");
    let l2_grad = l2_grad.expect("layer2 grad missing");

    assert_close(l2_grad[0], 1.0, 1e-5, "l2 grad[0]");
    assert_close(l2_grad[1], 2.0, 1e-5, "l2 grad[1]");

    assert_close(l1_grad[0], 1.0, 1e-5, "l1 grad[0]");
    assert_close(l1_grad[1], 2.0, 1e-5, "l1 grad[1]");
    assert_close(l1_grad[2], 1.0, 1e-5, "l1 grad[2]");
    assert_close(l1_grad[3], 2.0, 1e-5, "l1 grad[3]");
}

#[test]
fn test_seq_freeze_unfreeze_and_zero_grad() {
    let backend = Arc::new(CpuBackend::new());

    let mut layer = linear(1, 1).bias(true).build(&backend).unwrap();
    set_param(&mut layer.weight, &backend, &[2.0], &[1, 1]);
    set_param(layer.bias.as_mut().unwrap(), &backend, &[0.5], &[1]);

    let mut model: Seq<B, D> = Seq::new().push(layer);

    // Freeze: no grads should be populated
    model.freeze();
    let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);
    let x = Tensor::<B, D>::from_slice(&backend, &[3.0], &[1, 1]).unwrap();
    let y = model.forward(&ctx, ctx.input(&x)).unwrap();

    let mut dummy = Parameter::new(Tensor::<B, D>::from_slice(&backend, &[0.0], &[1, 1]).unwrap());
    let y_with_dummy = y.add(&ctx.param(&dummy)).unwrap();
    let loss = y_with_dummy.sum(None, false).unwrap();

    let mut params: Vec<&mut Parameter<B, D>> = model.params_mut();
    params.push(&mut dummy);
    ctx.backward(&loss, &mut params).unwrap();

    for p in model.params() {
        assert!(p.grad().is_none(), "Frozen param should have no grad");
    }
    assert!(dummy.grad().is_some(), "Dummy param should receive grad");

    // Unfreeze: grads should flow
    model.unfreeze();
    model.zero_grad();
    let ctx2 = Context::<B, D, Grad<B, D>>::grad(&backend);
    let y2 = model.forward(&ctx2, ctx2.input(&x)).unwrap();
    let loss2 = y2.sum(None, false).unwrap();
    ctx2.backward(&loss2, &mut model.params_mut()).unwrap();

    let params = model.params();
    assert_eq!(params.len(), 2); // weight and bias

    let mut w_grad: Option<Vec<f32>> = None;
    let mut b_grad: Option<Vec<f32>> = None;

    for p in params {
        if let Some(g) = p.grad() {
            let vals = tensor_to_vec(g);
            match p.name() {
                Some("weight") => w_grad = Some(vals),
                Some("bias") => b_grad = Some(vals),
                _ => panic!("unexpected param in linear"),
            }
        }
    }

    let w_grad = w_grad.expect("weight grad missing");
    let b_grad = b_grad.expect("bias grad missing");

    assert_close(w_grad[0], 3.0, 1e-5, "weight grad");
    assert_close(b_grad[0], 1.0, 1e-5, "bias grad");

    model.zero_grad();
    for p in model.params() {
        assert!(p.grad().is_none(), "zero_grad should clear gradients");
    }
}
