use std::{error::Error, sync::Arc};

use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn Error>> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0, 3.0, 2.0, 1.0], &[2, 2])?;
    println!("sum: {:?}", a.add(&b)?.to_vec()?);

    let lhs =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0], &[2, 3])?;
    let rhs =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;
    let product = lhs.matmul(&rhs)?;
    println!("matmul: {:?}", product.to_vec()?);

    let bias = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.5, -0.5], &[1, 2])?;
    let activated = product.add(&bias)?.contiguous()?;
    println!("add(matmul, bias): {:?}", activated.to_vec()?);

    Ok(())
}
