use std::error::Error;

use bolt_core::runtime::Runtime;
use bolt_cpu::{CpuBackend, CpuRuntimeBuilderExt};

fn main() -> Result<(), Box<dyn Error>> {
    let runtime = Runtime::builder().with_cpu()?.build()?;

    let a = runtime
        .tensor_from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0])?
        .try_into_tensor::<CpuBackend, f32>()?;
    let b = runtime
        .tensor_from_slice(&[2, 2], &[4.0f32, 3.0, 2.0, 1.0])?
        .try_into_tensor::<CpuBackend, f32>()?;
    println!("sum: {:?}", a.add(&b)?.to_vec()?);

    let lhs = runtime
        .tensor_from_slice(&[2, 3], &[1.0f32, 0.0, -1.0, 2.0, 1.0, 0.0])?
        .try_into_tensor::<CpuBackend, f32>()?;
    let rhs = runtime
        .tensor_from_slice(&[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?
        .try_into_tensor::<CpuBackend, f32>()?;
    let product = lhs.matmul(&rhs)?;
    println!("matmul: {:?}", product.to_vec()?);

    let bias = runtime
        .tensor_from_slice(&[1, 2], &[0.5f32, -0.5])?
        .try_into_tensor::<CpuBackend, f32>()?;
    let activated = product.add(&bias)?.contiguous()?;
    println!("add(matmul, bias): {:?}", activated.to_vec()?);

    Ok(())
}
