use std::ops::Add;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::{ConcreteShape, broadcast_shapes},
};

use super::super::allocator::CpuAllocator;
use super::super::layout_utils::{expand_strides, linear_to_indices, offset_from_strides};
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait AddKernel: NativeType {
    fn add_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "add not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub fn add<D>(
    lhs: CpuTensorView<'_, D>,
    rhs: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + Add<Output = D>,
{
    let lhs_shape = lhs.layout.shape();
    let rhs_shape = rhs.layout.shape();
    let shape = broadcast_shapes(lhs_shape, rhs_shape)?;
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    let lhs_strides = expand_strides(lhs.layout, &shape)?;
    let rhs_strides = expand_strides(rhs.layout, &shape)?;
    let lhs_data = lhs.storage.as_uninit_slice();
    let rhs_data = rhs.storage.as_uninit_slice();
    let lhs_offset = lhs.layout.offset_elements(D::DTYPE);
    let rhs_offset = rhs.layout.offset_elements(D::DTYPE);

    let mut coords = vec![0usize; shape.len()];
    for i in 0..numel {
        linear_to_indices(i, &shape, &mut coords);
        let lhs_idx = lhs_offset + offset_from_strides(&coords, &lhs_strides);
        let rhs_idx = rhs_offset + offset_from_strides(&coords, &rhs_strides);
        let lhs_val = unsafe { lhs_data[lhs_idx as usize].assume_init() };
        let rhs_val = unsafe { rhs_data[rhs_idx as usize].assume_init() };
        out_slice[i].write(lhs_val + rhs_val);
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(&shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl AddKernel for f32 {
    fn add_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        add(lhs, rhs, alloc)
    }
}

impl AddKernel for f64 {
    fn add_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        add(lhs, rhs, alloc)
    }
}

impl AddKernel for i32 {
    fn add_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        add(lhs, rhs, alloc)
    }
}
