use std::ops::Mul;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::can_use_fast_path_binary;

pub trait MulKernel: NativeType {
    fn mul_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "mul not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn mul<D>(
    lhs: CpuTensorView<'_, D>,
    rhs: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + Mul<Output = D>,
{
    let (lhs_layout, rhs_layout) = Layout::broadcast_binary(lhs.layout, rhs.layout)?;
    let shape = lhs_layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    let lhs_data = lhs.storage.as_uninit_slice();
    let rhs_data = rhs.storage.as_uninit_slice();

    if can_use_fast_path_binary(&lhs, &rhs, &lhs_layout, &rhs_layout) {
        for (dst, (&lhs_val, &rhs_val)) in out_slice
            .iter_mut()
            .zip(lhs_data.iter().zip(rhs_data.iter()))
        {
            let lhs_val = unsafe { lhs_val.assume_init() };
            let rhs_val = unsafe { rhs_val.assume_init() };
            dst.write(lhs_val * rhs_val);
        }
    } else {
        let lhs_iter = lhs_layout.iter_element_indices(D::DTYPE)?;
        let rhs_iter = rhs_layout.iter_element_indices(D::DTYPE)?;
        for (dst, (lhs_idx, rhs_idx)) in out_slice.iter_mut().zip(lhs_iter.zip(rhs_iter)) {
            let lhs_val = unsafe { lhs_data[lhs_idx].assume_init() };
            let rhs_val = unsafe { rhs_data[rhs_idx].assume_init() };
            dst.write(lhs_val * rhs_val);
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl MulKernel for f32 {
    fn mul_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul(lhs, rhs, alloc)
    }
}

impl MulKernel for f64 {
    fn mul_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul(lhs, rhs, alloc)
    }
}

impl MulKernel for i32 {
    fn mul_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul(lhs, rhs, alloc)
    }
}
