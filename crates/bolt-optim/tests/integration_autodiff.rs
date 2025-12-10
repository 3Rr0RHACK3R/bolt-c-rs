use std::sync::Arc;

use bolt_autodiff::{Autodiff, Parameter};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_optim::{Error, Sgd};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Opt = Sgd<CpuBackend, f32>;

#[test]
fn step_errors_on_missing_gradient() -> TestResult {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut p = Parameter::with_name(Tensor::from_slice(&base, &[1.0f32], &[1])?, "p");
    let mut q = Parameter::with_name(Tensor::from_slice(&base, &[2.0f32], &[1])?, "q");

    let mut opt: Opt = Opt::builder().learning_rate(0.1).init(&[&p, &q])?;

    autodiff.with_tape(|tape| {
        let pv = tape.param(&mut p);
        let loss = pv.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut p])
    })?;

    let err = opt.step(&mut [&mut p, &mut q]).unwrap_err();
    match err {
        Error::MissingGradient { param_id, .. } => assert_eq!(param_id, q.id()),
        other => panic!("unexpected error: {other:?}"),
    }

    Ok(())
}

#[test]
fn step_dedupes_repeated_params() -> TestResult {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut p = Parameter::with_name(Tensor::from_slice(&base, &[1.0f32], &[1])?, "p");
    let mut opt: Opt = Opt::builder().learning_rate(0.5).weight_decay(0.0).init(&[&p])?;

    autodiff.with_tape(|tape| {
        let pv = tape.param(&mut p);
        let loss = pv.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut p])
    })?;

    let before = p.value().to_vec()?;

    let mut dup_params: Vec<&mut Parameter<CpuBackend, f32>> = {
        let ptr: *mut Parameter<CpuBackend, f32> = &mut p;
        // SAFETY: we intentionally alias the same parameter to verify deduplication logic.
        vec![unsafe { &mut *ptr }, unsafe { &mut *ptr }]
    };

    opt.step(&mut dup_params)?;
    let after = p.value().to_vec()?;

    let expected = before[0] - 0.5 * 1.0;
    assert!((after[0] - expected).abs() < 1e-6, "dedupe failed: {after:?} vs {expected}");

    Ok(())
}

#[test]
fn weight_decay_applies_l2_penalty() -> TestResult {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let mut p = Parameter::with_name(Tensor::from_slice(&base, &[2.0f32], &[1])?, "p");
    let mut opt: Opt = Opt::builder()
        .learning_rate(1.0)
        .weight_decay(0.5)
        .init(&[&p])?;

    autodiff.with_tape(|tape| {
        let pv = tape.param(&mut p);
        let loss = pv.sum(None, false)?;
        tape.backward_into_params(&loss, &mut [&mut p])
    })?;

    opt.step(&mut [&mut p])?;
    let after = p.value().to_vec()?;

    // grad=1, decay=0.5*value=1.0 => effective grad 2.0, lr=1 => value should hit 0.0
    assert!((after[0] - 0.0).abs() < 1e-6, "weight decay not applied: {after:?}");

    Ok(())
}
