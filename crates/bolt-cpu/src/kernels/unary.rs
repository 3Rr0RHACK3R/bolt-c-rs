use bolt_core::{
    Operation,
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::{DType, NativeType},
    error::{Error, Result},
    op::{ExpOp, NegOp, ReluOp},
    tensor::Tensor,
};

use crate::kernels::common::{
    downcast_cpu, linear_to_indices, offset_from_strides, typed_storage, typed_storage_mut,
};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    register_unary::<f32, NegOp, _>(dispatcher, DType::F32, |v: f32| -v)?;
    register_unary::<f64, NegOp, _>(dispatcher, DType::F64, |v: f64| -v)?;
    register_unary::<i32, NegOp, _>(dispatcher, DType::I32, |v: i32| -v)?;

    register_unary::<f32, ExpOp, _>(dispatcher, DType::F32, |v: f32| v.exp())?;
    register_unary::<f64, ExpOp, _>(dispatcher, DType::F64, |v: f64| v.exp())?;

    register_unary::<f32, ReluOp, _>(dispatcher, DType::F32, |v: f32| v.max(0.0))?;
    register_unary::<f64, ReluOp, _>(dispatcher, DType::F64, |v: f64| v.max(0.0))?;

    Ok(())
}

fn register_unary<T, O, F>(dispatcher: &mut Dispatcher, dtype: DType, func: F) -> Result<()>
where
    T: NativeType,
    O: Operation,
    F: Fn(T) -> T + Send + Sync + Copy + 'static,
{
    dispatcher.register_operation::<O, _>(
        DeviceKind::Cpu,
        dtype,
        KernelLayoutReq::GeneralStrided,
        move |inputs, op| unary_kernel::<T, O, F>(inputs, op, func),
    )
}

fn unary_kernel<T, O, F>(inputs: &[Tensor], _op: &O, func: F) -> Result<Vec<Tensor>>
where
    T: NativeType,
    O: Operation,
    F: Fn(T) -> T + Copy,
{
    if inputs.len() != 1 {
        return Err(Error::Device(format!("{:?} expects 1 input", O::KIND)));
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
    let strides = input.strides().to_vec();

    in_cell.with_read(|in_buf| {
        out_cell.with_write(|out_buf| {
            let in_vals = typed_storage::<T>(in_buf);
            let out_vals = typed_storage_mut::<T>(out_buf);
            let mut coords = vec![0usize; input.shape().len()];
            let total = input.numel();
            let base = input.layout().offset_elements(input.dtype());
            let out_base = output.layout().offset_elements(output.dtype());
            for linear in 0..total {
                linear_to_indices(linear, input.shape(), &mut coords);
                let src_idx = base + offset_from_strides(&coords, &strides);
                let dst_idx = out_base + linear as isize;
                out_vals[dst_idx as usize] = func(in_vals[src_idx as usize]);
            }
            Ok(())
        })
    })?;

    Ok(vec![output])
}
