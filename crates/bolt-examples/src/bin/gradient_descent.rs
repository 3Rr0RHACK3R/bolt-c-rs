use std::sync::Arc;

use bolt_autodiff::{Autodiff, Parameter};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lr = 0.1f32;
    let steps = 80usize;

    // Create backend
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Autodiff::wrap(cpu_backend.clone());

    let lr_tensor = Tensor::from_slice(&cpu_backend, &[lr], &[])?;

    // Create training data on base backend
    let x_data = Tensor::from_slice(&cpu_backend, &[0.0f32, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2])?;
    let y_data = Tensor::from_slice(&cpu_backend, &[1.0f32, 3.0, 5.0, 7.0], &[4, 1])?;

    // Create a learnable parameter using Parameter API
    let w_data = Tensor::from_slice(&cpu_backend, &[0.0f32, 0.0], &[2, 1])?;
    let mut w_param = Parameter::with_name(w_data, "weights");

    for step in 0..steps {
        // Use with_tape for scoped gradient tracking
        autodiff.with_tape(|tape| {
            // Wrap inputs with tape.input()
            let x = tape.input(&x_data);
            let y = tape.input(&y_data);

            // Track parameter on tape
            let w = tape.param(&w_param);

            // Forward pass
            let y_pred = x.matmul(&w)?;
            let diff = y_pred.sub(&y)?;
            let loss = diff.mul(&diff)?.mean(None, false)?;

            // Backward pass - compute gradients into parameter
            tape.backward_into_params(&loss, &mut [&mut w_param])
        })?;

        // Get gradient and update weights
        if let Some(dw) = w_param.grad() {
            let scaled = dw.mul(&lr_tensor)?;
            let new_w = w_param.tensor().sub(&scaled)?;
            *w_param.tensor_mut() = new_w;
        }

        // Print progress
        if (step + 1) % 10 == 0 || step == 0 {
            let w_vec = w_param.tensor().to_vec()?;
            println!(
                "step {:>3}: w=[{:.4}, {:.4}]",
                step + 1,
                w_vec[0],
                w_vec[1],
            );
        }

        // Zero gradients for next iteration
        w_param.zero_grad();
    }

    println!("\nFinal weights: {:?}", w_param.tensor().to_vec()?);
    println!("Expected: [2.0, 1.0] (y = 2x + 1)");

    Ok(())
}
