use bolt_autodiff::{Autodiff, Error, Parameter, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(approx_eq(*a, *e, eps), "got {}, expected {} (eps={})", a, e, eps);
    }
}

#[test]
fn backward_into_params_sets_grad() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut w = Parameter::with_name(Tensor::from_slice(&base, &[2.0_f32], &[1])?, "w");

    autodiff.with_tape(|tape| {
        let x = Tensor::from_slice(&base, &[3.0_f32], &[1])?;
        let xv = tape.input(&x);
        let wv = tape.param(&mut w);
        let loss = xv.mul(&wv)?.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut w])
    })?;

    let grad = w.grad().expect("grad for w").to_vec()?;
    assert_vec_approx_eq(&grad, &[3.0], 1e-6);

    Ok(())
}

#[test]
fn missing_param_in_tape_errors() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut used = Parameter::with_name(Tensor::from_slice(&base, &[1.0_f32], &[1])?, "used");
    let mut missing = Parameter::with_name(Tensor::from_slice(&base, &[2.0_f32], &[1])?, "missing");

    let err = autodiff
        .with_tape(|tape| {
            let x = Tensor::from_slice(&base, &[4.0_f32], &[1])?;
            let used_v = tape.param(&mut used);
            let loss = used_v.mul(&tape.input(&x))?.sum(None, false)?;
            tape.backward_into_params(&loss, &mut [&mut used, &mut missing])
        })
        .unwrap_err();

    match err {
        Error::ParamNotInTape { param_id, .. } => assert_eq!(param_id, missing.id()),
        other => panic!("unexpected error: {other:?}"),
    }

    Ok(())
}

#[test]
fn backward_param_grads_missing_param_errors() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut used = Parameter::with_name(Tensor::from_slice(&base, &[1.0_f32], &[1])?, "used");
    let missing = Parameter::with_name(Tensor::from_slice(&base, &[2.0_f32], &[1])?, "missing");

    let err = match autodiff.with_tape(|tape| {
        let x = Tensor::from_slice(&base, &[3.0_f32], &[1])?;
        let used_v = tape.param(&mut used);
        let loss = used_v.mul(&tape.input(&x))?.sum(None, false)?;
        tape.backward_param_grads(&loss, &[&used, &missing])
    }) {
        Ok(_) => panic!("expected ParamNotInTape error"),
        Err(e) => e,
    };

    match err {
        Error::ParamNotInTape { param_id, .. } => assert_eq!(param_id, missing.id()),
        other => panic!("unexpected error: {other:?}"),
    }

    Ok(())
}

#[test]
fn backward_clears_unused_param_grad() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut w = Parameter::with_name(Tensor::from_slice(&base, &[1.0_f32], &[1])?, "w");
    let mut b = Parameter::with_name(Tensor::from_slice(&base, &[1.0_f32], &[1])?, "b");

    autodiff.with_tape(|tape| {
        let x = Tensor::from_slice(&base, &[2.0_f32], &[1])?;
        let xv = tape.input(&x);
        let wv = tape.param(&mut w);
        let bv = tape.param(&mut b);
        let loss = wv.add(&bv)?.mul(&xv)?.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut w, &mut b])
    })?;

    assert!(w.grad().is_some());
    assert!(b.grad().is_some());

    autodiff.with_tape(|tape| {
        let x = Tensor::from_slice(&base, &[3.0_f32], &[1])?;
        let xv = tape.input(&x);
        let wv = tape.param(&mut w);
        let _ = tape.param(&mut b);
        let loss = wv.mul(&xv)?.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut w, &mut b])
    })?;

    assert!(w.grad().is_some());
    assert!(b.grad().is_none());

    Ok(())
}

#[test]
fn backward_param_grads_does_not_mutate_params() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut w = Parameter::with_name(Tensor::from_slice(&base, &[2.0_f32], &[1])?, "w");

    let grads = autodiff.with_tape(|tape| {
        let x = Tensor::from_slice(&base, &[5.0_f32], &[1])?;
        let xv = tape.input(&x);
        let wv = tape.param(&mut w);
        let loss = wv.mul(&xv)?.sum(None, false)?;
        tape.backward_param_grads(&loss, &[&w])
    })?;

    assert!(w.grad().is_none());

    let g = grads.get(&w).expect("grad for w").to_vec()?;
    assert_vec_approx_eq(&g, &[5.0], 1e-6);

    Ok(())
}

#[test]
fn duplicate_param_calls_reuse_handle() -> Result<()> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut w = Parameter::with_name(Tensor::from_slice(&base, &[1.0_f32], &[1])?, "w");

    let grads = autodiff.with_tape(|tape| {
        let wv1 = tape.param(&mut w);
        let wv2 = tape.param(&mut w);
        let loss = wv1.add(&wv2)?.sum(None, false)?;
        tape.backward_param_grads(&loss, &[&w, &w])
    })?;

    assert!(w.grad().is_none());
    let grad = grads.get(&w).expect("grad for w").to_vec()?;
    assert_vec_approx_eq(&grad, &[2.0], 1e-6);

    Ok(())
}
