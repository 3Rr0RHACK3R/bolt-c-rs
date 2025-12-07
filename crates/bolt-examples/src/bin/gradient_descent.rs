use std::sync::Arc;

use bolt_autodiff::{Autodiff, AutodiffTensorExt};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lr = 0.1f32;
    let steps = 80usize;

    let x_host = vec![0.0f32, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
    let y_host = vec![1.0f32, 3.0, 5.0, 7.0];

    // Create backend and wrap with autodiff
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let lr_tensor = Tensor::from_slice(&cpu_backend, &[lr], &[])?;
    let mut w_data = Tensor::from_slice(&cpu_backend, &[0.0f32, 0.0], &[2, 1])?;

    for step in 0..steps {
        // Begin gradient tracking scope
        let _ctx = autodiff.begin_grad();

        // Create parameter tensor that requires gradients
        let w = Tensor::from_slice(&autodiff, &w_data.to_vec()?, &[2, 1])?.requires_grad();

        // Forward pass
        let x = Tensor::from_slice(&autodiff, &x_host, &[4, 2])?;
        let y = Tensor::from_slice(&autodiff, &y_host, &[4, 1])?;

        let y_pred = x.matmul(&w)?;
        let diff = y_pred.sub(&y)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        // Backward pass - returns gradients on CPU backend
        let grads = loss.backward()?;
        let dw = grads.wrt(&w).unwrap();

        // Update weights (gradients are on cpu_backend, same as w_data)
        let scaled = dw.mul(&lr_tensor)?;
        w_data = w_data.sub(&scaled)?;

        // Print progress
        if (step + 1) % 10 == 0 || step == 0 {
            let loss_value = loss.detach().item()?;
            let w_vec = w_data.to_vec()?;
            println!(
                "step {:>3}: loss={:.4}, w=[{:.4}, {:.4}]",
                step + 1,
                loss_value,
                w_vec[0],
                w_vec[1],
            );
        }

        // _ctx dropped here, graph is cleared automatically
    }

    println!("\nFinal weights: {:?}", w_data.to_vec()?);

    Ok(())
}
