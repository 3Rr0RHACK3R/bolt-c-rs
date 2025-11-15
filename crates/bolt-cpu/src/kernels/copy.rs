use std::sync::Arc;

use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::{DType, NativeType},
    error::{Error, Result},
    op::{OpKey, OpKind},
    tensor::Tensor,
};

use crate::kernels::common::{
    downcast_cpu, linear_to_indices, offset_from_strides, typed_storage, typed_storage_mut,
};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    register_copy::<f32>(dispatcher, DType::F32)?;
    register_copy::<f64>(dispatcher, DType::F64)?;
    register_copy::<i32>(dispatcher, DType::I32)?;
    Ok(())
}

fn register_copy<T>(dispatcher: &mut Dispatcher, dtype: DType) -> Result<()>
where
    T: NativeType,
{
    let key = OpKey {
        op: OpKind::Copy,
        device: DeviceKind::Cpu,
        dtype,
    };
    dispatcher.register(
        key,
        KernelLayoutReq::GeneralStrided,
        Arc::new(|inputs, _| copy_kernel::<T>(inputs)),
    )
}

fn copy_kernel<T>(inputs: &[Tensor]) -> Result<Vec<Tensor>>
where
    T: NativeType,
{
    if inputs.len() != 1 {
        return Err(Error::Device("copy expects 1 input".into()));
    }
    let input = &inputs[0];
    if input.dtype() != T::DTYPE {
        return Err(Error::DTypeMismatch {
            lhs: input.dtype(),
            rhs: T::DTYPE,
        });
    }
    let device = input.device()?;
    let cpu = downcast_cpu(&device)?;
    let runtime = input.runtime();
    let output = runtime.allocate_uninit(input.device_kind(), input.shape(), input.dtype())?;
    let in_cell = cpu.buffer_cell(input.buffer_id())?;
    let out_cell = cpu.buffer_cell(output.buffer_id())?;
    let in_strides = input.strides().to_vec();

    in_cell.with_read(|in_buf| {
        out_cell.with_write(|out_buf| {
            let in_vals = typed_storage::<T>(in_buf);
            let out_vals = typed_storage_mut::<T>(out_buf);
            let mut coords = vec![0usize; input.shape().len()];
            let total = input.numel();
            let in_base = input.layout().offset_elements(input.dtype());
            let out_base = output.layout().offset_elements(output.dtype());
            for linear in 0..total {
                linear_to_indices(linear, input.shape(), &mut coords);
                let src_idx = in_base + offset_from_strides(&coords, &in_strides);
                let dst_idx = out_base + linear as isize;
                out_vals[dst_idx as usize] = in_vals[src_idx as usize];
            }
            Ok(())
        })
    })?;

    Ok(vec![output])
}
