#![cfg(any(test, feature = "test-kernels"))]

use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, Result},
    op::FillOp,
    tensor::Tensor,
};

use super::common::downcast_cpu;

pub fn register_test_poison_kernel(dispatcher: &mut Dispatcher) -> Result<()> {
    dispatcher.register_operation::<FillOp, _>(
        DeviceKind::Cpu,
        DType::F32,
        KernelLayoutReq::GeneralStrided,
        poison_kernel,
    )
}

fn poison_kernel(inputs: &[Tensor], _op: &FillOp) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(Error::Device("test poison kernel expects 1 input".into()));
    }
    let tensor = &inputs[0];
    let device = tensor.device()?;
    let cpu = downcast_cpu(&device)?;
    let cell = cpu.buffer_cell(tensor.buffer_id())?;

    match cell.with_write(|_| -> Result<Vec<Tensor>> {
        panic!("intentional cpu poison kernel panic for tests");
    }) {
        Ok(_) => unreachable!("poison kernel must panic"),
        Err(err) => Err(err),
    }
}
