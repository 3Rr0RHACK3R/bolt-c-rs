use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::{DType, NativeType},
    error::{Error, Result},
    op::{AddOp, DivOp, MulOp, SubOp},
    shape::broadcast_shapes,
    tensor::Tensor,
};

use crate::kernels::common::{
    downcast_cpu, linear_to_indices, offset_from_strides, typed_storage, typed_storage_mut,
};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    register_binary::<f32, AddOp, _>(dispatcher, DType::F32, "add", |a, b| a + b)?;
    register_binary::<f64, AddOp, _>(dispatcher, DType::F64, "add", |a, b| a + b)?;
    register_binary::<i32, AddOp, _>(dispatcher, DType::I32, "add", |a, b| a + b)?;

    register_binary::<f32, SubOp, _>(dispatcher, DType::F32, "sub", |a, b| a - b)?;
    register_binary::<f64, SubOp, _>(dispatcher, DType::F64, "sub", |a, b| a - b)?;
    register_binary::<i32, SubOp, _>(dispatcher, DType::I32, "sub", |a, b| a - b)?;

    register_binary::<f32, MulOp, _>(dispatcher, DType::F32, "mul", |a, b| a * b)?;
    register_binary::<f64, MulOp, _>(dispatcher, DType::F64, "mul", |a, b| a * b)?;
    register_binary::<i32, MulOp, _>(dispatcher, DType::I32, "mul", |a, b| a * b)?;

    register_binary::<f32, DivOp, _>(dispatcher, DType::F32, "div", |a, b| a / b)?;
    register_binary::<f64, DivOp, _>(dispatcher, DType::F64, "div", |a, b| a / b)?;
    register_binary::<i32, DivOp, _>(dispatcher, DType::I32, "div", |a, b| a / b)?;

    Ok(())
}

fn register_binary<T, O, F>(
    dispatcher: &mut Dispatcher,
    dtype: DType,
    name: &'static str,
    func: F,
) -> Result<()>
where
    T: NativeType,
    O: bolt_core::op::Operation,
    F: Fn(T, T) -> T + Send + Sync + Copy + 'static,
{
    dispatcher.register_operation::<O, _>(
        DeviceKind::Cpu,
        dtype,
        KernelLayoutReq::GeneralStrided,
        move |inputs, _| binary_kernel::<T, F>(inputs, name, func),
    )
}

fn binary_kernel<T, F>(inputs: &[Tensor], name: &'static str, func: F) -> Result<Vec<Tensor>>
where
    T: NativeType,
    F: Fn(T, T) -> T + Copy,
{
    if inputs.len() != 2 {
        return Err(Error::Device(format!("{name} expects 2 inputs")));
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    if lhs.dtype() != T::DTYPE || rhs.dtype() != T::DTYPE {
        return Err(Error::DTypeMismatch {
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    let plan = BroadcastPlan::new(lhs, rhs)?;
    let device = lhs.device()?;
    let cpu = downcast_cpu(&device)?;
    let runtime = lhs.runtime();
    let output = runtime.allocate_uninit(lhs.device_kind(), &plan.shape, lhs.dtype())?;
    let lhs_cell = cpu.buffer_cell(lhs.buffer_id())?;
    let rhs_cell = cpu.buffer_cell(rhs.buffer_id())?;
    let out_cell = cpu.buffer_cell(output.buffer_id())?;

    lhs_cell.with_read(|lhs_buf| {
        rhs_cell.with_read(|rhs_buf| {
            out_cell.with_write(|out_buf| {
                let lhs_vals = typed_storage::<T>(lhs_buf);
                let rhs_vals = typed_storage::<T>(rhs_buf);
                let out_vals = typed_storage_mut::<T>(out_buf);
                let mut coords = vec![0usize; plan.shape.len()];
                let total: usize = plan.shape.iter().product();
                let out_base = output.layout().offset_elements(output.dtype());
                for linear in 0..total {
                    linear_to_indices(linear, &plan.shape, &mut coords);
                    let lhs_idx = plan.lhs_base + offset_from_strides(&coords, &plan.lhs_strides);
                    let rhs_idx = plan.rhs_base + offset_from_strides(&coords, &plan.rhs_strides);
                    let dst = out_base + linear as isize;
                    out_vals[dst as usize] =
                        func(lhs_vals[lhs_idx as usize], rhs_vals[rhs_idx as usize]);
                }
                Ok(())
            })
        })
    })?;

    Ok(vec![output])
}

struct BroadcastPlan {
    shape: Vec<usize>,
    lhs_strides: Vec<isize>,
    rhs_strides: Vec<isize>,
    lhs_base: isize,
    rhs_base: isize,
}

impl BroadcastPlan {
    fn new(lhs: &Tensor, rhs: &Tensor) -> Result<Self> {
        let shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
        let lhs_strides = expand_strides(lhs, &shape)?;
        let rhs_strides = expand_strides(rhs, &shape)?;
        Ok(Self {
            lhs_base: lhs.layout().offset_elements(lhs.dtype()),
            rhs_base: rhs.layout().offset_elements(rhs.dtype()),
            shape,
            lhs_strides,
            rhs_strides,
        })
    }
}

fn expand_strides(tensor: &Tensor, target: &[usize]) -> Result<Vec<isize>> {
    let in_shape = tensor.shape();
    let in_strides = tensor.strides();
    let mut result = vec![0isize; target.len()];
    let mut in_idx = in_shape.len();
    for out_idx in (0..target.len()).rev() {
        let out_dim = target[out_idx];
        if in_idx == 0 {
            result[out_idx] = 0;
            continue;
        }
        in_idx -= 1;
        let in_dim = in_shape[in_idx];
        let stride = in_strides[in_idx];
        if in_dim == out_dim {
            result[out_idx] = stride;
        } else if in_dim == 1 && out_dim >= 1 {
            result[out_idx] = 0;
        } else if out_dim == 1 {
            result[out_idx] = stride;
        } else {
            return Err(Error::ShapeMismatch {
                lhs: in_shape.to_vec(),
                rhs: target.to_vec(),
            });
        }
    }
    Ok(result)
}
