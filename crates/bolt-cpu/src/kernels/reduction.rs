use std::ops::Add;

use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, Result},
    op::SumOp,
    shape::{ConcreteShape, canonical_axes, reduced_shape},
    tensor::Tensor,
};

use crate::kernels::common::{
    Numeric, downcast_cpu, linear_to_indices, offset_from_strides, typed_storage, typed_storage_mut,
};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    register_sum::<f32>(dispatcher, DType::F32)?;
    register_sum::<f64>(dispatcher, DType::F64)?;
    register_sum::<i32>(dispatcher, DType::I32)?;
    Ok(())
}

fn register_sum<T>(dispatcher: &mut Dispatcher, dtype: DType) -> Result<()>
where
    T: Numeric + Add<Output = T>,
{
    dispatcher.register_operation::<SumOp, _>(
        DeviceKind::Cpu,
        dtype,
        KernelLayoutReq::GeneralStrided,
        |inputs, op| sum_kernel::<T>(inputs, op),
    )
}

fn sum_kernel<T>(inputs: &[Tensor], op: &SumOp) -> Result<Vec<Tensor>>
where
    T: Numeric + Add<Output = T>,
{
    if inputs.len() != 1 {
        return Err(Error::Device("sum expects 1 input".into()));
    }
    let input = &inputs[0];
    if input.dtype() != T::DTYPE {
        return Err(Error::DTypeMismatch {
            lhs: input.dtype(),
            rhs: T::DTYPE,
        });
    }
    let rank = input.shape().len();
    let axes_vec = canonical_axes(&op.axes, rank)?;
    let device = input.device()?;
    let cpu = downcast_cpu(&device)?;
    let out_shape = reduced_shape(input.shape(), &axes_vec)?;
    let runtime = input.runtime();
    let output = runtime.allocate_uninit(input.device_kind(), &out_shape, input.dtype())?;
    let in_cell = cpu.buffer_cell(input.buffer_id())?;
    let out_cell = cpu.buffer_cell(output.buffer_id())?;
    let axes_mask = axes_mask(&axes_vec, rank);
    let out_strides = ConcreteShape::from_slice(&out_shape)?.contiguous_strides();

    in_cell.with_read(|in_buf| {
        out_cell.with_write(|out_buf| {
            let in_vals = typed_storage::<T>(in_buf);
            let out_vals = typed_storage_mut::<T>(out_buf);
            let out_base = output.layout().offset_elements(output.dtype());
            let out_total = output.numel();
            for idx in 0..out_total {
                out_vals[(out_base + idx as isize) as usize] = T::zero();
            }
            let mut coords = vec![0usize; rank];
            let in_strides = input.strides().to_vec();
            let in_base = input.layout().offset_elements(input.dtype());
            for linear in 0..input.numel() {
                linear_to_indices(linear, input.shape(), &mut coords);
                let mut out_index = out_base;
                let mut out_axis = 0;
                for axis in 0..rank {
                    if axes_mask[axis] {
                        continue;
                    }
                    let stride = out_strides[out_axis];
                    out_index += coords[axis] as isize * stride;
                    out_axis += 1;
                }
                let in_idx = in_base + offset_from_strides(&coords, &in_strides);
                let dst = out_index as usize;
                out_vals[dst] = out_vals[dst] + in_vals[in_idx as usize];
            }
            Ok(())
        })
    })?;

    Ok(vec![output])
}

fn axes_mask(axes: &[usize], rank: usize) -> Vec<bool> {
    let mut mask = vec![false; rank];
    for &axis in axes {
        mask[axis] = true;
    }
    mask
}
