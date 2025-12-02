use std::sync::Arc;

use bolt_autodiff::Graph;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let lr = 0.1f32;
    let steps = 80usize;

    let x_host = vec![0.0f32, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
    let y_host = vec![1.0f32, 3.0, 5.0, 7.0];
    let x_tensor = Tensor::from_slice(&backend, &x_host, &[4, 2])?;
    let y_tensor = Tensor::from_slice(&backend, &y_host, &[4, 1])?;
    let lr_tensor = Tensor::from_slice(&backend, &[lr], &[])?;
    let mut w_tensor = Tensor::from_slice(&backend, &[0.0f32, 0.0], &[2, 1])?;

    for step in 0..steps {
        let graph = Graph::<CpuBackend, f32>::new(backend.clone());

        let x = graph.constant(&x_tensor);
        let y = graph.constant(&y_tensor);
        let w = graph.variable(&w_tensor);

        let y_pred = x.matmul(&w)?;
        let diff = y_pred.sub(&y)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        let grads = graph.backward(&loss)?;
        let dw = grads.wrt(&w).unwrap();

        let scaled = dw.mul(&lr_tensor)?;
        w_tensor = w_tensor.sub(&scaled)?;

        if (step + 1) % 10 == 0 || step == 0 {
            let loss_value = loss.tensor()?.item()?;
            println!(
                "step {:>3}: loss={:.4}, w={:?}",
                step + 1,
                loss_value,
                w_tensor,
            );
        }
    }

    Ok(())
}
