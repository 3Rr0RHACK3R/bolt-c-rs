use std::{
    ops::{AddAssign, Mul},
    sync::Arc,
};

use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, Result},
    op::{OpAttrs, OpKey, OpKind},
    tensor::Tensor,
};

use crate::kernels::common::{Numeric, contiguous_slice, contiguous_slice_mut, downcast_cpu};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    register_matmul::<f32>(dispatcher, DType::F32)?;
    register_matmul::<f64>(dispatcher, DType::F64)?;
    Ok(())
}

fn register_matmul<T>(dispatcher: &mut Dispatcher, dtype: DType) -> Result<()>
where
    T: Numeric + Copy + AddAssign + Mul<Output = T>,
{
    let key = OpKey {
        op: OpKind::MatMul,
        device: DeviceKind::Cpu,
        dtype,
    };
    dispatcher.register(
        key,
        KernelLayoutReq::Contiguous,
        Arc::new(|inputs, attrs| matmul_kernel::<T>(inputs, attrs)),
    )
}

fn matmul_kernel<T>(inputs: &[Tensor], _: &OpAttrs) -> Result<Vec<Tensor>>
where
    T: Numeric + Copy + AddAssign + Mul<Output = T>,
{
    if inputs.len() != 2 {
        return Err(Error::Device("matmul expects 2 inputs".into()));
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    if lhs.dtype() != T::DTYPE || rhs.dtype() != T::DTYPE {
        return Err(Error::DTypeMismatch {
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    let device = lhs.device()?;
    let cpu = downcast_cpu(&device)?;
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let runtime = lhs.runtime();
    let output = runtime.allocate_uninit(lhs.device_kind(), &[m, n], lhs.dtype())?;
    let lhs_cell = cpu.buffer_cell(lhs.buffer_id())?;
    let rhs_cell = cpu.buffer_cell(rhs.buffer_id())?;
    let out_cell = cpu.buffer_cell(output.buffer_id())?;

    lhs_cell.with_read(|lhs_buf| {
        rhs_cell.with_read(|rhs_buf| {
            out_cell.with_write(|out_buf| {
                let lhs_vals = contiguous_slice::<T>(lhs, lhs_buf)?;
                let rhs_vals = contiguous_slice::<T>(rhs, rhs_buf)?;
                let out_vals = contiguous_slice_mut::<T>(&output, out_buf)?;
                for value in out_vals.iter_mut() {
                    *value = T::zero();
                }
                for i in 0..m {
                    for kk in 0..k {
                        let lhs_val = lhs_vals[i * k + kk];
                        let rhs_row = &rhs_vals[kk * n..(kk + 1) * n];
                        let out_row = &mut out_vals[i * n..(i + 1) * n];
                        for j in 0..n {
                            out_row[j] += lhs_val * rhs_row[j];
                        }
                    }
                }
                Ok(())
            })
        })
    })?;

    Ok(vec![output])
}
