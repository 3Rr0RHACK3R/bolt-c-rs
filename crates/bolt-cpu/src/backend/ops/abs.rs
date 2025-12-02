use std::ops::Neg;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait AbsKernel: NativeType {
    fn abs_kernel(
        _view: CpuTensorView<'_, Self>,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "abs not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn abs<D>(
    view: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + PartialOrd + Neg<Output = D>,
{
    let shape = view.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;
    let view_data = view.storage.as_uninit_slice();
    let zero = D::default();

    if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
        for (dst, &val) in out_data.iter_mut().zip(view_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(if val < zero { -val } else { val });
        }
    } else {
        for (dst, idx) in out_data.iter_mut().zip(view.layout.iter_element_indices(D::DTYPE)?) {
            let val = unsafe { view_data[idx].assume_init() };
            dst.write(if val < zero { -val } else { val });
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl AbsKernel for f32 {
    fn abs_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(view, allocator)
    }
}

impl AbsKernel for f64 {
    fn abs_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(view, allocator)
    }
}

impl AbsKernel for i32 {
    fn abs_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(view, allocator)
    }
}
