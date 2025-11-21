use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::{NativeType, ToF32},
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::layout_utils::{linear_to_indices, offset_from_strides};
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait MeanKernel: NativeType + ToF32 {
    fn mean_f32_kernel(
        input: CpuTensorView<'_, Self>,
        alloc_f32: &CpuAllocator<f32>,
    ) -> Result<TensorParts<CpuStorage<f32>>>;
}

fn reduce_to_mean_f32<D>(
    input: CpuTensorView<'_, D>,
    alloc_f32: &CpuAllocator<f32>,
) -> Result<TensorParts<CpuStorage<f32>>>
where
    D: NativeType + ToF32 + Copy,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Err(Error::OpError("cannot compute mean of empty tensor".into()));
    }

    let strides = input.layout.strides();
    let data = input.storage.as_slice();
    let offset = input.layout.offset_elements(D::DTYPE);

    let mut sum: f32 = 0.0;

    // Optimization: if contiguous and no offset, iterate slice directly
    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        if data.len() < numel {
            return Err(Error::OpError("storage smaller than shape requires".into()));
        }
        for x in data.iter().take(numel) {
            sum += x.to_f32();
        }
    } else {
        let mut coords = vec![0usize; shape.len()];
        for i in 0..numel {
            linear_to_indices(i, shape, &mut coords);
            let idx = offset + offset_from_strides(&coords, strides);
            sum += data[idx as usize].to_f32();
        }
    }

    let mean = sum / (numel as f32);

    let mut out_storage = alloc_f32.allocate(1)?;
    out_storage.try_as_mut_slice()?[0] = mean;

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&[1])?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl MeanKernel for f32 {
    fn mean_f32_kernel(
        input: CpuTensorView<'_, Self>,
        alloc_f32: &CpuAllocator<f32>,
    ) -> Result<TensorParts<CpuStorage<f32>>> {
        reduce_to_mean_f32(input, alloc_f32)
    }
}

impl MeanKernel for f64 {
    fn mean_f32_kernel(
        input: CpuTensorView<'_, Self>,
        alloc_f32: &CpuAllocator<f32>,
    ) -> Result<TensorParts<CpuStorage<f32>>> {
        reduce_to_mean_f32(input, alloc_f32)
    }
}

impl MeanKernel for i32 {
    fn mean_f32_kernel(
        input: CpuTensorView<'_, Self>,
        alloc_f32: &CpuAllocator<f32>,
    ) -> Result<TensorParts<CpuStorage<f32>>> {
        reduce_to_mean_f32(input, alloc_f32)
    }
}
