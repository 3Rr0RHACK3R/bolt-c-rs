use std::sync::Arc;

use bolt_autodiff::{Autodiff, AutodiffTensorExt, Function, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, eps),
            "element {} differs: got {}, expected {} (eps={})",
            i,
            a,
            e,
            eps
        );
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Test: Single input -> single output (Swish)
// Swish(x) = x * sigmoid(x)
// d/dx Swish(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

struct Swish;

#[derive(Default, Clone)]
struct SwishCtx {
    input: Option<Tensor<CpuBackend, f32>>,
    sigmoid: Option<Tensor<CpuBackend, f32>>,
}

impl Function<CpuBackend, f32> for Swish {
    type Ctx = SwishCtx;

    fn forward(ctx: &mut Self::Ctx, inputs: &[&Tensor<CpuBackend, f32>]) -> Result<Vec<Tensor<CpuBackend, f32>>> {
        let x = inputs[0];
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;
        let one = Tensor::ones(&x.backend(), x.shape())?;
        let denom = one.add(&exp_neg_x)?;
        let sigmoid = Tensor::ones(&x.backend(), x.shape())?.div(&denom)?;
        let output = x.mul(&sigmoid)?;
        
        ctx.input = Some(x.clone()); // x
        ctx.sigmoid = Some(sigmoid); // sigmoid(x)
        
        Ok(vec![output])
    }

    fn backward(
        ctx: &Self::Ctx,
        grad_outputs: &[Option<&Tensor<CpuBackend, f32>>],
    ) -> Result<Vec<Option<Tensor<CpuBackend, f32>>>> {
        let grad_output = grad_outputs[0].unwrap();
        let x = ctx.input.as_ref().unwrap();
        let sigmoid = ctx.sigmoid.as_ref().unwrap();
        
        let one = Tensor::ones(&x.backend(), x.shape())?;
        let one_minus_sigmoid = one.sub(sigmoid)?;
        let x_times_deriv = x.mul(&one_minus_sigmoid)?;
        let bracket = Tensor::ones(&x.backend(), x.shape())?.add(&x_times_deriv)?;
        let local_grad = sigmoid.mul(&bracket)?;
        let grad_input = grad_output.mul(&local_grad)?;
        
        Ok(vec![Some(grad_input)])
    }
}

#[test]
fn test_custom_function_single_input_single_output() -> Result<()> {
    let cpu = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu.clone()));
    
    let _ctx = autodiff.begin_grad();
    
    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, -1.0, 0.0], &[4])?.requires_grad();
    
    let outputs = Swish::apply(&[&x])?;
    let y = &outputs[0];
    
    let loss = y.sum(None, false)?;
    let grads = loss.backward()?;
    
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;
    
    // grad = sigmoid * (1 + x * (1 - sigmoid))
    let calculate_swish_grad = |x_val: f32| -> f32 {
        let sigmoid_x = sigmoid(x_val);
        sigmoid_x * (1.0 + x_val * (1.0 - sigmoid_x))
    };

    let grad_1 = calculate_swish_grad(1.0);
    let grad_2 = calculate_swish_grad(2.0);
    let grad_neg1 = calculate_swish_grad(-1.0);
    let grad_0 = calculate_swish_grad(0.0);
    
    assert_vec_approx_eq(&dx, &[grad_1, grad_2, grad_neg1, grad_0], 1e-5);
    
    Ok(())
}

// Test: Multi-input -> single output (weighted sum)
// f(a, b, w) = a * w + b * (1 - w)
// df/da = w, df/db = 1 - w, df/dw = a - b

struct WeightedSum;

#[derive(Default, Clone)]
struct WeightedSumCtx {
    a: Option<Tensor<CpuBackend, f32>>,
    b: Option<Tensor<CpuBackend, f32>>,
    w: Option<Tensor<CpuBackend, f32>>,
}

impl Function<CpuBackend, f32> for WeightedSum {
    type Ctx = WeightedSumCtx;

    fn forward(ctx: &mut Self::Ctx, inputs: &[&Tensor<CpuBackend, f32>]) -> Result<Vec<Tensor<CpuBackend, f32>>> {
        let a = inputs[0];
        let b = inputs[1];
        let w = inputs[2];
        
        let one = Tensor::ones(&a.backend(), a.shape())?;
        let one_minus_w = one.sub(w)?;
        let aw = a.mul(w)?;
        let b_1_w = b.mul(&one_minus_w)?;
        let output = aw.add(&b_1_w)?;
        
        ctx.a = Some(a.clone());
        ctx.b = Some(b.clone());
        ctx.w = Some(w.clone());
        
        Ok(vec![output])
    }

    fn backward(
        ctx: &Self::Ctx,
        grad_outputs: &[Option<&Tensor<CpuBackend, f32>>],
    ) -> Result<Vec<Option<Tensor<CpuBackend, f32>>>> {
        let grad_output = grad_outputs[0].unwrap();
        let a = ctx.a.as_ref().unwrap();
        let b = ctx.b.as_ref().unwrap();
        let w = ctx.w.as_ref().unwrap();
        
        let grad_a = grad_output.mul(w)?;
        
        let one = Tensor::ones(&a.backend(), a.shape())?;
        let grad_b = grad_output.mul(&one.sub(w)?)?;
        
        let a_minus_b = a.sub(b)?;
        let grad_w = grad_output.mul(&a_minus_b)?;
        
        Ok(vec![Some(grad_a), Some(grad_b), Some(grad_w)])
    }
}

#[test]
fn test_custom_function_multi_input_single_output() -> Result<()> {
    let cpu = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu.clone()));
    
    let _ctx = autodiff.begin_grad();
    
    let a = Tensor::from_slice(&autodiff, &[3.0_f32, 4.0], &[2])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?.requires_grad();
    let w = Tensor::from_slice(&autodiff, &[0.7_f32, 0.3], &[2])?.requires_grad();
    
    let outputs = WeightedSum::apply(&[&a, &b, &w])?;
    let y = &outputs[0];
    
    let loss = y.sum(None, false)?;
    let grads = loss.backward()?;
    
    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    
    // da = w = [0.7, 0.3]
    // db = 1 - w = [0.3, 0.7]
    // dw = a - b = [2.0, 2.0]
    assert_vec_approx_eq(&da, &[0.7, 0.3], 1e-5);
    assert_vec_approx_eq(&db, &[0.3, 0.7], 1e-5);
    assert_vec_approx_eq(&dw, &[2.0, 2.0], 1e-5);
    
    Ok(())
}

// Test: No gradient context
#[test]
fn test_custom_function_no_grad() -> Result<()> {
    let cpu = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu.clone()));
    
    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?;
    
    let outputs = Swish::apply(&[&x])?;
    let y = &outputs[0];
    
    let y_vec = y.detach().to_vec()?;
    
    // Swish(1) = 1 * sigmoid(1) ≈ 0.7311
    // Swish(2) = 2 * sigmoid(2) ≈ 1.7616
    let expected_1 = 1.0 * sigmoid(1.0);
    let expected_2 = 2.0 * sigmoid(2.0);
    
    assert_vec_approx_eq(&y_vec, &[expected_1, expected_2], 1e-5);
    
    Ok(())
}
