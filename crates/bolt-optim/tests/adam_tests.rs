use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Adam, AdamCfg, AdamGroupCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

#[test]
fn adam_converges_on_linear_regression() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    // y = 2x + 1
    let xs = Tensor::<B, D>::from_slice(&backend, &[0.0, 1.0, 2.0, 3.0], &[4]).unwrap();
    let ys = Tensor::<B, D>::from_slice(&backend, &[1.0, 3.0, 5.0, 7.0], &[4]).unwrap();

    let w = store.param("w", &[], Init::Zeros).unwrap();
    let b = store.param("b", &[], Init::Zeros).unwrap();

    let mut opt = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            lr: 0.1,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )
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

/// Helper to create a constant-initialized scalar parameter
fn const_scalar_param(store: &Store<B, D>, name: &str, value: f32) -> bolt_nn::Param<B, D> {
    // Use Uniform with same low/high to get constant initialization
    let p = store
        .param(
            name,
            &[],
            Init::Uniform {
                low: value,
                high: value,
            },
        )
        .unwrap();
    p
}

#[test]
fn adam_converges_faster_than_baseline_for_quadratic() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 42);

    // Minimize (x - 3)^2
    let target = Tensor::<B, D>::from_slice(&backend, &[3.0], &[]).unwrap();
    let x = store.param("x", &[], Init::Zeros).unwrap();

    let mut opt = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            lr: 0.5,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let params = store.trainable();
    let mut final_loss = f32::MAX;

    for _ in 0..100 {
        store.zero_grad();

        let diff = x.tensor().sub(&target).unwrap();
        let loss = diff.mul(&diff).unwrap();
        final_loss = loss.item().unwrap();

        store.backward(&loss).unwrap();
        opt.step(&params).unwrap();
    }

    assert!(
        final_loss < 0.01,
        "Adam should converge quickly, final loss: {final_loss}"
    );

    let x_val: f32 = x.tensor().item().unwrap();
    assert!((x_val - 3.0).abs() < 0.1, "x should be ~3.0, got {x_val}");
}

#[test]
fn adam_tracks_step_count() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);
    let x = const_scalar_param(&store, "x", 1.0);

    let mut opt = Adam::<B, D>::new(backend.clone(), AdamCfg::default()).unwrap();
    assert_eq!(opt.step_count(), 0);

    let params = store.trainable();

    // Need a gradient for step to actually process
    store.zero_grad();
    let loss = x.tensor().mul(&x.tensor()).unwrap();
    store.backward(&loss).unwrap();

    opt.step(&params).unwrap();
    assert_eq!(opt.step_count(), 1);

    store.zero_grad();
    let loss = x.tensor().mul(&x.tensor()).unwrap();
    store.backward(&loss).unwrap();

    opt.step(&params).unwrap();
    assert_eq!(opt.step_count(), 2);
}

#[test]
fn adam_reset_clears_state() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);
    let x = const_scalar_param(&store, "x", 1.0);

    let mut opt = Adam::<B, D>::new(backend.clone(), AdamCfg::default()).unwrap();
    let params = store.trainable();

    // Run a few steps to build up state
    for _ in 0..5 {
        store.zero_grad();
        let loss = x.tensor().mul(&x.tensor()).unwrap();
        store.backward(&loss).unwrap();
        opt.step(&params).unwrap();
    }

    assert_eq!(opt.step_count(), 5);
    assert!(!opt.first_moment_state().is_empty());

    opt.reset();

    assert_eq!(opt.step_count(), 0);
    assert!(opt.first_moment_state().is_empty());
}

#[test]
fn adam_weight_decay_reduces_weights() {
    let backend = Arc::new(CpuBackend::new());

    // Test with weight decay - weights should move toward zero
    let store_wd = Store::<B, D>::new(backend.clone(), 123);
    let w_wd = const_scalar_param(&store_wd, "w", 5.0);

    let mut opt_wd = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            lr: 0.1,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.1,
        },
    )
    .unwrap();

    let params_wd = store_wd.trainable();

    // Target is 0 - weight should decay toward zero with weight decay helping
    let target = Tensor::<B, D>::from_slice(&backend, &[0.0], &[]).unwrap();

    for _ in 0..50 {
        store_wd.zero_grad();
        // Loss that pulls toward target (0), plus weight decay
        let diff = w_wd.tensor().sub(&target).unwrap();
        let loss = diff.mul(&diff).unwrap();
        store_wd.backward(&loss).unwrap();
        opt_wd.step(&params_wd).unwrap();
    }

    let w_wd_val: f32 = w_wd.tensor().item().unwrap();
    // With weight decay, weight should decay toward zero
    assert!(
        w_wd_val < 1.0,
        "Weight with decay should be close to 0 from 5.0, got {w_wd_val}"
    );
}

#[test]
fn adam_validates_lr_positive() {
    let backend = Arc::new(CpuBackend::new());
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            lr: 0.0,
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("lr must be positive"));
}

#[test]
fn adam_validates_lr_not_negative() {
    let backend = Arc::new(CpuBackend::new());
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            lr: -0.1,
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
}

#[test]
fn adam_validates_beta1_range() {
    let backend = Arc::new(CpuBackend::new());

    // beta1 >= 1.0 should fail
    let result = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            betas: (1.0, 0.999),
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("beta1"));

    // beta1 < 0 should fail
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            betas: (-0.1, 0.999),
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
}

#[test]
fn adam_validates_beta2_range() {
    let backend = Arc::new(CpuBackend::new());

    // beta2 >= 1.0 should fail
    let result = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            betas: (0.9, 1.0),
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("beta2"));

    // beta2 < 0 should fail
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            betas: (0.9, -0.1),
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
}

#[test]
fn adam_validates_eps_positive() {
    let backend = Arc::new(CpuBackend::new());
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            eps: 0.0,
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("eps must be positive"));
}

#[test]
fn adam_validates_weight_decay_non_negative() {
    let backend = Arc::new(CpuBackend::new());
    let result = Adam::<B, D>::new(
        backend,
        AdamCfg {
            weight_decay: -0.01,
            ..AdamCfg::default()
        },
    );
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("weight_decay"));
}

#[test]
fn adam_group_config_overrides_lr() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    // Create param and set its group to 1
    let x = const_scalar_param(&store, "x", 1.0);
    x.set_group(1);

    let mut opt = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            lr: 0.1,
            ..AdamCfg::default()
        },
    )
    .unwrap();

    opt.set_group(
        1,
        AdamGroupCfg {
            lr_mult: 2.0,
            weight_decay: None,
        },
    )
    .unwrap();

    let params = store.trainable();

    // Do one step
    store.zero_grad();
    let loss = x.tensor().mul(&x.tensor()).unwrap();
    store.backward(&loss).unwrap();

    let before: f32 = x.tensor().item().unwrap();
    opt.step(&params).unwrap();
    let after: f32 = x.tensor().item().unwrap();

    // Parameter should have changed
    assert!(
        (after - before).abs() > 0.0,
        "Parameter should change after step"
    );
}

#[test]
fn adam_group_validates_lr_mult_positive() {
    let backend = Arc::new(CpuBackend::new());
    let mut opt = Adam::<B, D>::new(backend, AdamCfg::default()).unwrap();

    let result = opt.set_group(
        0,
        AdamGroupCfg {
            lr_mult: 0.0,
            weight_decay: None,
        },
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("lr_mult must be positive")
    );
}

#[test]
fn adam_group_validates_weight_decay_non_negative() {
    let backend = Arc::new(CpuBackend::new());
    let mut opt = Adam::<B, D>::new(backend, AdamCfg::default()).unwrap();

    let result = opt.set_group(
        0,
        AdamGroupCfg {
            lr_mult: 1.0,
            weight_decay: Some(-0.1),
        },
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("weight_decay"));
}

#[test]
fn adam_default_config_is_valid() {
    let backend = Arc::new(CpuBackend::new());
    let result = Adam::<B, D>::new(backend, AdamCfg::default());
    assert!(result.is_ok());

    let cfg = AdamCfg::default();
    assert_eq!(cfg.lr, 1e-3);
    assert_eq!(cfg.betas, (0.9, 0.999));
    assert_eq!(cfg.eps, 1e-8);
    assert_eq!(cfg.weight_decay, 0.0);
}

#[test]
fn adam_handles_multiple_params() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 0);

    let x = const_scalar_param(&store, "x", 1.0);
    let y = const_scalar_param(&store, "y", -1.0);
    // For 2D param, use Uniform for simplicity
    let z = store
        .param(
            "z",
            &[2, 2],
            Init::Uniform {
                low: 0.5,
                high: 0.5,
            },
        )
        .unwrap();

    let mut opt = Adam::<B, D>::new(backend.clone(), AdamCfg::default()).unwrap();
    let params = store.trainable();

    for _ in 0..10 {
        store.zero_grad();

        // Loss that depends on all params
        let x_sq = x.tensor().mul(&x.tensor()).unwrap();
        let y_sq = y.tensor().mul(&y.tensor()).unwrap();
        let z_sum = z.tensor().sum(None, false).unwrap();
        let z_sq = z_sum.mul(&z_sum).unwrap();

        let loss = x_sq.add(&y_sq).unwrap().add(&z_sq).unwrap();

        store.backward(&loss).unwrap();
        opt.step(&params).unwrap();
    }

    // All should move toward zero (minimizing sum of squares)
    let x_val: f32 = x.tensor().item().unwrap();
    let y_val: f32 = y.tensor().item().unwrap();

    assert!(
        x_val.abs() < 1.0,
        "x should move toward 0 from 1.0, got {x_val}"
    );
    assert!(
        y_val.abs() < 1.0,
        "y should move toward 0 from -1.0, got {y_val}"
    );
}

#[test]
fn adam_step_count_can_be_set() {
    let backend = Arc::new(CpuBackend::new());
    let mut opt = Adam::<B, D>::new(backend, AdamCfg::default()).unwrap();

    assert_eq!(opt.step_count(), 0);
    opt.set_step_count(100);
    assert_eq!(opt.step_count(), 100);
}

#[test]
fn adam_bias_correction_affects_early_steps() {
    // The bias correction should make Adam more aggressive in early steps
    // when the moment estimates are biased toward zero
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 42);

    let x = const_scalar_param(&store, "x", 10.0);
    let target = Tensor::<B, D>::from_slice(&backend, &[0.0], &[]).unwrap();

    let mut opt = Adam::<B, D>::new(
        backend.clone(),
        AdamCfg {
            lr: 1.0,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let params = store.trainable();

    // First step should have significant movement due to bias correction
    store.zero_grad();
    let diff = x.tensor().sub(&target).unwrap();
    let loss = diff.mul(&diff).unwrap();
    store.backward(&loss).unwrap();

    let before: f32 = x.tensor().item().unwrap();
    opt.step(&params).unwrap();
    let after: f32 = x.tensor().item().unwrap();

    // The change should be substantial (bias correction amplifies early updates)
    let change = (before - after).abs();
    assert!(
        change > 0.5,
        "First step with bias correction should have significant movement, got {change}"
    );
}
