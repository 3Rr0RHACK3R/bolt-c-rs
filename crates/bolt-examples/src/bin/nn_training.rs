use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::Linear;
use bolt_nn::{Module, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

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

        let output = layer.forward(x_data.clone(), true)?;
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
    let pred = layer.forward(test_x, false)?;
    println!("pred(x=5)={:.4}", pred.to_vec()?[0]);

    Ok(())
}
