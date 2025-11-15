use std::error::Error;

use bolt_core::Runtime;
use bolt_cpu::CpuRuntimeBuilderExt;

fn main() -> Result<(), Box<dyn Error>> {
    let runtime = Runtime::builder().with_cpu()?.build()?;

    let a = runtime.tensor_from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0])?;
    let b = runtime.tensor_from_slice(&[2, 2], &[4.0f32, 3.0, 2.0, 1.0])?;
    let c = a.add(&b)?;
    println!("sum: {:?}", c.to_vec::<f32>()?);

    let lhs = runtime.tensor_from_slice(&[2, 3], &[1.0f32, 0.0, -1.0, 2.0, 1.0, 0.0])?;
    let rhs = runtime.tensor_from_slice(&[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let product = lhs.matmul(&rhs)?;
    println!("matmul: {:?}", product.to_vec::<f32>()?);

    let bias = runtime.tensor_from_slice(&[1, 2], &[0.5f32, -0.5])?;
    let activated = product.add(&bias)?.relu()?;
    println!("relu(add(matmul, bias)): {:?}", activated.to_vec::<f32>()?);

    let loss = activated.sum()?.to_vec::<f32>()?.pop().unwrap();
    println!("sum reduction: {loss}");

    Ok(())
}
