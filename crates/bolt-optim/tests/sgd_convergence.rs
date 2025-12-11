use std::sync::Arc;

use bolt_autodiff::{Autodiff, Parameter};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_optim::Sgd;

type Opt = Sgd<CpuBackend, f32>;

#[test]
fn sgd_converges_on_linear_regression() -> Result<(), Box<dyn std::error::Error>> {
    let base = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(base.clone());

    let xs = Tensor::from_slice(&base, &[0.0f32, 1.0, 2.0, 3.0], &[4])?;
    let ys = Tensor::from_slice(&base, &[1.0f32, 3.0, 5.0, 7.0], &[4])?;

    let mut w = Parameter::with_name(Tensor::from_slice(&base, &[0.0f32], &[])?, "w");
    let mut b = Parameter::with_name(Tensor::from_slice(&base, &[0.0f32], &[])?, "b");

    let mut opt: Opt = Opt::builder()
        .learning_rate(0.1)
        .momentum(0.9)
        .init(&[&w, &b])?;

    for _ in 0..200 {
        autodiff.with_tape(|tape| {
            let x = tape.input(&xs);
            let y = tape.input(&ys);
            let wv = tape.param(&mut w);
            let bv = tape.param(&mut b);
            let y_pred = x.mul(&wv)?.add(&bv)?;
            let diff = y_pred.sub(&y)?;
            let loss = diff.mul(&diff)?.mean(None, false)?;
            tape.backward_into_params(&loss, &mut [&mut w, &mut b])
        })?;

        opt.step(&mut [&mut w, &mut b])?;
    }

    let w_val: f32 = w.tensor().item()?;
    let b_val: f32 = b.tensor().item()?;

    assert!((w_val - 2.0).abs() < 0.05, "w too far: {w_val}");
    assert!((b_val - 1.0).abs() < 0.05, "b too far: {b_val}");

    Ok(())
}
