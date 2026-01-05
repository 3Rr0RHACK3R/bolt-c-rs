use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn missing_gradient_does_not_update_parameter() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    let p = store.param("p", &[1], Init::Zeros).unwrap();
    let q = store.param("q", &[1], Init::Zeros).unwrap();

    p.set_tensor(Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap())
        .unwrap();
    q.set_tensor(Tensor::<B, D>::from_slice(&backend, &[2.0], &[1]).unwrap())
        .unwrap();

    p.set_grad(Some(
        Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap(),
    ));
    q.set_grad(None);

    let mut opt = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 1.0,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    opt.step(&store.trainable()).unwrap();

    let p_after = p.tensor().to_vec().unwrap();
    let q_after = q.tensor().to_vec().unwrap();

    assert_eq!(p_after, vec![0.0]);
    assert_eq!(q_after, vec![2.0]);
}

#[test]
fn weight_decay_applies_l2_penalty() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_tensor(Tensor::<B, D>::from_slice(&backend, &[2.0], &[1]).unwrap())
        .unwrap();
    p.set_grad(Some(
        Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap(),
    ));

    let mut opt = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 1.0,
            momentum: 0.0,
            weight_decay: 0.5,
        },
    )
    .unwrap();

    opt.step(&store.trainable()).unwrap();
    let after = p.tensor().to_vec().unwrap();

    assert!((after[0] - 0.0).abs() < 1e-6, "weight decay not applied");
}
