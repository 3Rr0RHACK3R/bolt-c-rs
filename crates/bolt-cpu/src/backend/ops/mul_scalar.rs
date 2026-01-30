use std::ops::Mul;

use bolt_core::{
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    StorageAllocator, TensorParts,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait MulScalarKernel: NativeType {
    fn mul_scalar_kernel(
        _view: CpuTensorView<'_, Self>,
        _scalar: Self,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "mul_scalar not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn mul_scalar<D>(
    view: CpuTensorView<'_, D>,
    scalar: D,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Mul<Output = D>,
{
    let shape = view.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;
    let view_data = view.storage.as_uninit_slice();

    if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
        for (dst, &val) in out_data.iter_mut().zip(view_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(val * scalar);
        }
    } else {
        for (dst, idx) in out_data
            .iter_mut()
            .zip(view.layout.iter_element_indices(D::DTYPE)?)
        {
            let val = unsafe { view_data[idx].assume_init() };
            dst.write(val * scalar);
        }
    }

    let layout = Layout::contiguous(shape.clone());
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl MulScalarKernel for f32 {
    fn mul_scalar_kernel(
        view: CpuTensorView<'_, Self>,
        scalar: Self,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul_scalar(view, scalar, allocator)
    }
}

impl MulScalarKernel for f64 {
    fn mul_scalar_kernel(
        view: CpuTensorView<'_, Self>,
        scalar: Self,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul_scalar(view, scalar, allocator)
    }
}

impl MulScalarKernel for i32 {
    fn mul_scalar_kernel(
        view: CpuTensorView<'_, Self>,
        scalar: Self,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        mul_scalar(view, scalar, allocator)
    }
}
