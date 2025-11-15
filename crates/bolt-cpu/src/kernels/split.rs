use std::sync::Arc;

use bolt_core::{
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, Result},
    op::{OpAttrs, OpKey, OpKind, SplitSpecAttrs},
    tensor::Tensor,
};

pub fn register(dispatcher: &mut Dispatcher) -> Result<()> {
    for &dtype in &[DType::F32, DType::F64, DType::I32] {
        let key = OpKey {
            op: OpKind::Split,
            device: DeviceKind::Cpu,
            dtype,
        };
        dispatcher.register(
            key,
            KernelLayoutReq::GeneralStrided,
            Arc::new(|inputs, attrs| split_kernel(inputs, attrs)),
        )?;
    }
    Ok(())
}

fn split_kernel(inputs: &[Tensor], attrs: &OpAttrs) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(Error::Device("split expects 1 input tensor".into()));
    }
    let input = &inputs[0];
    let split_attrs = attrs
        .split_attrs()
        .ok_or_else(|| Error::Device("split kernel requires split attrs".into()))?;
    let axis = split_attrs.axis;
    if axis >= input.shape().len() {
        return Err(Error::invalid_shape(format!(
            "axis {axis} out of bounds for rank {}",
            input.shape().len()
        )));
    }
    let dim = input.shape()[axis];
    let mut outputs = Vec::new();
    match &split_attrs.spec {
        SplitSpecAttrs::ChunkSize { size } => {
            let mut start = 0;
            while start < dim {
                let end = (start + size).min(dim);
                outputs.push(input.slice(axis, start, end, 1)?);
                start = end;
            }
        }
        SplitSpecAttrs::Sections(sections) => {
            let mut start = 0;
            for &section in sections {
                let end = start + section;
                if end > dim {
                    return Err(Error::invalid_shape("sections exceed axis length"));
                }
                outputs.push(input.slice(axis, start, end, 1)?);
                start = end;
            }
            if start != dim {
                return Err(Error::invalid_shape("sections do not cover axis length"));
            }
        }
    }
    if outputs.is_empty() {
        return Err(Error::Device("split kernel produced no outputs".into()));
    }
    Ok(outputs)
}
