use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn sgd_converges_on_linear_regression() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let xs = Tensor::<B, D>::from_slice(&backend, &[0.0, 1.0, 2.0, 3.0], &[4]).unwrap();
    let ys = Tensor::<B, D>::from_slice(&backend, &[1.0, 3.0, 5.0, 7.0], &[4]).unwrap();

    // params initialized to 0 here, and during training will be updated
    let w = store.param("w", &[], Init::Zeros).unwrap();
    let b = store.param("b", &[], Init::Zeros).unwrap();

    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })
    .unwrap();

    let params = store.trainable();

    for _ in 0..200 {
        store.zero_grad();

        let wv = w.tensor().broadcast_to(xs.shape().as_slice()).unwrap();
        let bv = b.tensor().broadcast_to(xs.shape().as_slice()).unwrap();
        let y_pred = xs.mul(&wv).unwrap().add(&bv).unwrap();
        let diff = y_pred.sub(&ys).unwrap();
        let loss = diff.mul(&diff).unwrap().mean(None, false).unwrap();

        store.backward(&loss).unwrap();
        opt.step(&params).unwrap();
    }

    let w_val: f32 = w.tensor().item().unwrap();
    let b_val: f32 = b.tensor().item().unwrap();

    assert!((w_val - 2.0).abs() < 0.05, "w too far: {w_val}");
    assert!((b_val - 1.0).abs() < 0.05, "b too far: {b_val}");
}
