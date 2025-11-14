use std::sync::Arc;

use bolt_core::{
    device::Device,
    dtype::NativeType,
    error::{Error, Result},
    tensor::Tensor,
};
use bytemuck::{cast_slice, cast_slice_mut};

use crate::device::CpuDevice;

pub fn downcast_cpu(device: &Arc<dyn Device>) -> Result<&CpuDevice> {
    device
        .as_ref()
        .as_any()
        .downcast_ref::<CpuDevice>()
        .ok_or_else(|| Error::Device("expected CpuDevice".into()))
}

pub fn linear_to_indices(mut index: usize, shape: &[usize], coords: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        coords[axis] = index % dim;
        index /= dim;
    }
}

pub fn offset_from_strides(indices: &[usize], strides: &[isize]) -> isize {
    indices
        .iter()
        .zip(strides.iter())
        .fold(0isize, |acc, (idx, stride)| acc + *idx as isize * *stride)
}

pub fn typed_storage<'a, T: NativeType>(buffer: &'a [u8]) -> &'a [T] {
    cast_slice(buffer)
}

pub fn typed_storage_mut<'a, T: NativeType>(buffer: &'a mut [u8]) -> &'a mut [T] {
    cast_slice_mut(buffer)
}

pub fn contiguous_slice<'a, T: NativeType>(tensor: &Tensor, buffer: &'a [u8]) -> Result<&'a [T]> {
    if !tensor.is_contiguous() {
        return Err(Error::invalid_shape("tensor must be contiguous"));
    }
    let start = tensor.layout().offset_elements(tensor.dtype()) as usize;
    let end = start + tensor.numel();
    Ok(&typed_storage::<T>(buffer)[start..end])
}

pub fn contiguous_slice_mut<'a, T: NativeType>(
    tensor: &Tensor,
    buffer: &'a mut [u8],
) -> Result<&'a mut [T]> {
    if !tensor.is_contiguous() {
        return Err(Error::invalid_shape("tensor must be contiguous"));
    }
    let start = tensor.layout().offset_elements(tensor.dtype()) as usize;
    let end = start + tensor.numel();
    Ok(&mut typed_storage_mut::<T>(buffer)[start..end])
}

pub trait Numeric: NativeType {
    fn zero() -> Self;
}

impl Numeric for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Numeric for i32 {
    fn zero() -> Self {
        0
    }
}
