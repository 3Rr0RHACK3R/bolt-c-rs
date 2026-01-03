use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn step_dedupes_duplicate_params_by_id() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_tensor(Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap())
        .unwrap();
    p.set_grad(Some(
        Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap(),
    ));

    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 1.0,
        momentum: 0.0,
        weight_decay: 0.0,
    })
    .unwrap();

    opt.step(&[p.clone(), p.clone()]).unwrap();

    let after = p.tensor().to_vec().unwrap();
    // The parameter "p" starts at 1.0 and has a gradient of 1.0.
    // The optimizer's lr (learning rate) is 1.0, momentum is 0.0, weight_decay is 0.0.
    // Calling opt.step with [p.clone(), p.clone()] passes two references to the same param.
    // The optimizer must deduplicate params by ID, so only a single update is applied:
    // new_value = old_value - lr * grad = 1.0 - 1.0 * 1.0 = 0.0
    // So after the step, p.tensor() should yield vec![0.0].
    assert_eq!(after, vec![0.0]);
}

#[test]
fn velocity_state_keyed_by_param_id() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_tensor(Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap())
        .unwrap();
    p.set_grad(Some(
        Tensor::<B, D>::from_slice(&backend, &[1.0], &[1]).unwrap(),
    ));

    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 1.0,
        momentum: 0.9,
        weight_decay: 0.0,
    })
    .unwrap();

    opt.step(&store.trainable()).unwrap();

    // Velocity keyed by ParamId
    assert!(opt.velocity_state().contains_key(&p.id()));
}
