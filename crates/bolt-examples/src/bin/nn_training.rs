use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Linear;
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_rng::RngKey;
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());

    let root_key = RngKey::from_seed(1337);
    let init_key = root_key.derive("init");
    let store = Store::<B, D>::new_with_init_key(backend.clone(), init_key);

    let layer = Linear::init(&store.sub("linear"), 1, 1, true)?;
    store.seal();

    let params = store.trainable();
    let mut opt = Sgd::<B, D>::new(SgdCfg {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    })?;

    let x_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4, 1])?;
    let y_data = Tensor::<B, D>::from_slice(&backend, &[3.0, 5.0, 7.0, 9.0], &[4, 1])?;

    for step in 0..200 {
        store.zero_grad();

        let step_key = root_key.derive("step").fold_in(step);
        let mut ctx = ForwardCtx::train_with_key(step_key);
        let output = layer.forward(x_data.clone(), &mut ctx)?;
        let diff = output.sub(&y_data)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        store.backward(&loss)?;
        opt.step(&params)?;

        if (step + 1) % 20 == 0 || step == 0 {
            let w = layer.weight.tensor().to_vec()?[0];
            let b = layer.bias.as_ref().unwrap().tensor().to_vec()?[0];
            println!("step {:>3}: w={:.4} b={:.4}", step + 1, w, b);
        }
    }

    let test_x = Tensor::<B, D>::from_slice(&backend, &[5.0], &[1, 1])?;
    let mut ctx = ForwardCtx::eval();
    let pred = layer.forward(test_x, &mut ctx)?;
    println!("pred(x=5)={:.4}", pred.to_vec()?[0]);

    Ok(())
}
