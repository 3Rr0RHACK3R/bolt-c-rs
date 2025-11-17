mod binary;
mod common;
mod copy;
mod matmul;
mod reduction;
mod split;
#[cfg(any(test, feature = "test-kernels"))]
mod test_poison;
mod unary;

#[cfg(any(test, feature = "test-kernels"))]
pub use test_poison::register_test_poison_kernel;

use bolt_core::{dispatcher::Dispatcher, error::Result};

pub fn register_cpu_kernels(dispatcher: &mut Dispatcher) -> Result<()> {
    binary::register(dispatcher)?;
    unary::register(dispatcher)?;
    reduction::register(dispatcher)?;
    matmul::register(dispatcher)?;
    copy::register(dispatcher)?;
    split::register(dispatcher)?;
    Ok(())
}
