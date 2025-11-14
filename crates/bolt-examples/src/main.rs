use std::{error::Error, sync::Arc};

use bolt_core::{Tensor, init_dispatcher};
use bolt_cpu::{CpuDevice, register_cpu_kernels};

fn main() -> Result<(), Box<dyn Error>> {
    init_dispatcher(|dispatcher| register_cpu_kernels(dispatcher))?;

    let device = Arc::new(CpuDevice::new());

    let a = Tensor::from_slice(device.clone(), &[2, 2], &[1.0f32, 2.0, 3.0, 4.0])?;
    let b = Tensor::from_slice(device.clone(), &[2, 2], &[4.0f32, 3.0, 2.0, 1.0])?;
    let c = a.add(&b)?;
    println!("sum: {:?}", c.to_vec::<f32>()?);

    let lhs = Tensor::from_slice(device.clone(), &[2, 3], &[1.0f32, 0.0, -1.0, 2.0, 1.0, 0.0])?;
    let rhs = Tensor::from_slice(device.clone(), &[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let product = lhs.matmul(&rhs)?;
    println!("matmul: {:?}", product.to_vec::<f32>()?);

    let bias = Tensor::from_slice(device.clone(), &[1, 2], &[0.5f32, -0.5])?;
    let activated = product.add(&bias)?.relu()?;
    println!("relu(add(matmul, bias)): {:?}", activated.to_vec::<f32>()?);

    let loss = activated.sum()?.to_vec::<f32>()?.pop().unwrap();
    println!("sum reduction: {loss}");

    Ok(())
}
