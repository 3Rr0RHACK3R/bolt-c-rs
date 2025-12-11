//! Example demonstrating the Unified Context API for training neural networks.
//!
//! This example shows how to:
//! - Use `Context::eval()` for inference
//! - Use `Context::grad()` for training with automatic differentiation
//! - Use `ctx.backward()` to compute and populate gradients
//! - Use the SGD optimizer from bolt-optim

use std::sync::Arc;

use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::linear;
use bolt_nn::{Context, Eval, Grad, Model};
use bolt_optim::Sgd;

type B = CpuBackend;
type D = f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let steps = 100;

    // Create training data: y = 2x + 1
    let x_data = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4, 1])?;
    let y_data = Tensor::<B, D>::from_slice(&backend, &[3.0, 5.0, 7.0, 9.0], &[4, 1])?;

    // Build the model - just needs backend, not context
    let mut layer: bolt_nn::layers::Linear<B, D> = linear(1, 1).build(&backend)?;

    // Initialize the SGD optimizer with layer parameters
    let mut optimizer = Sgd::<B, D>::builder()
        .learning_rate(0.1)
        .init(&layer.params())?;

    println!("Training a linear model to learn y = 2x + 1\n");

    // Training loop
    for step in 0..steps {
        // Create a fresh Grad context for each forward-backward pass
        let ctx = Context::<B, D, Grad<B, D>>::grad(&backend);

        // Forward pass
        let output = layer.forward(&ctx, ctx.input(&x_data))?;

        // Compute MSE loss
        let diff = output.sub(&ctx.input(&y_data))?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        // Backward pass - populates gradients on parameters
        ctx.backward(&loss, &mut layer.params_mut())?;

        // Optimizer step - updates weights using gradients
        optimizer.step(&mut layer.params_mut())?;

        // Zero gradients for next iteration
        for param in layer.params_mut() {
            param.zero_grad();
        }

        // Print progress
        if (step + 1) % 20 == 0 || step == 0 {
            let w = layer.weight.tensor().to_vec()?[0];
            let b = layer.bias.as_ref().unwrap().tensor().to_vec()?[0];
            println!("Step {:>3}: w = {:.4}, b = {:.4}", step + 1, w, b);
        }
    }

    // Final evaluation using Eval context
    let eval_ctx = Context::<B, D, Eval<B, D>>::eval(&backend);
    let test_x = Tensor::<B, D>::from_slice(&backend, &[5.0], &[1, 1])?;
    let pred = layer.forward(&eval_ctx, eval_ctx.input(&test_x))?;

    println!("\n--- Results ---");
    println!("Weight: {:.4}", layer.weight.tensor().to_vec()?[0]);
    println!("Bias:   {:.4}", layer.bias.as_ref().unwrap().tensor().to_vec()?[0]);
    println!("Expected: w=2.0, b=1.0");
    println!("\nPrediction for x=5: {:.4}", pred.to_vec()?[0]);
    println!("Expected: 11.0 (2*5 + 1)");

    Ok(())
}
