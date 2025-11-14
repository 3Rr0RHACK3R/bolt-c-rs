mod binary;
mod common;
mod copy;
mod matmul;
mod reduction;
mod unary;

use bolt_core::{dispatcher::Dispatcher, error::Result};

pub fn register_cpu_kernels(dispatcher: &mut Dispatcher) -> Result<()> {
    binary::register(dispatcher)?;
    unary::register(dispatcher)?;
    reduction::register(dispatcher)?;
    matmul::register(dispatcher)?;
    copy::register(dispatcher)?;
    Ok(())
}
