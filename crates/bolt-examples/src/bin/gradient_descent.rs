use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{Sgd, SgdCfg};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let x =
        Tensor::<B, D>::from_slice(&backend, &[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2])?;
    let y = Tensor::<B, D>::from_slice(&backend, &[1.0, 3.0, 5.0, 7.0], &[4, 1])?;

    let w = store.param("w", &[2, 1], Init::Zeros)?;
    let params = store.trainable();

    let mut opt = Sgd::<B, D>::new(
        backend.clone(),
        SgdCfg {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )?;

    for step in 0..80 {
        store.zero_grad();

        let y_pred = x.matmul(&w.tensor())?;
        let diff = y_pred.sub(&y)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        store.backward(&loss)?;
        opt.step(&params)?;

        if (step + 1) % 10 == 0 || step == 0 {
            let w_vec = w.tensor().to_vec()?;
            println!("step {:>3}: w=[{:.4}, {:.4}]", step + 1, w_vec[0], w_vec[1]);
        }
    }

    println!("final w: {:?}", w.tensor().to_vec()?);
    Ok(())
}
