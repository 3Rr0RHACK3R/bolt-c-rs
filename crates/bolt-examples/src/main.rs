use std::{error::Error, sync::Arc};

use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0f32, 3.0, 2.0, 1.0], &[2, 2])?;
    println!("sum: {:?}", a.add(&b)?.to_vec()?);

    let lhs = Tensor::from_slice(&backend, &[1.0f32, 0.0, -1.0, 2.0, 1.0, 0.0], &[2, 3])?;
    let rhs = Tensor::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;
    let product = lhs.matmul(&rhs)?; // shape: [2, 2]
    println!("matmul: {:?}", product.to_vec()?);

    let bias = Tensor::from_slice(&backend, &[0.5, -0.5], &[1, 2])?;
    // broadcasted add followed by reshape [2, 2] -> [4]
    let activated = product.add(&bias)?.reshape(&[4])?;
    println!("add(matmul, bias): {:?}", activated.to_vec()?);

    let first = activated.slice(0, 0, 1, 1)?;
    println!("shape: {:?}", first.shape());
    println!("add(matmul, bias)[0,0]: {}", first.item()?);

    Ok(())
}
