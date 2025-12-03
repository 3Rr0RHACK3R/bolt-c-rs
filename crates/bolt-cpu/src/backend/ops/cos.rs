use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::FloatType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait CosKernel: FloatType {
    fn cos_kernel(
        _view: CpuTensorView<'_, Self>,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "cos not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn cos<D>(
    view: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: FloatType + num_traits::Float,
{
    let shape = view.layout.shape();
    let numel = view.layout.num_elements();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;
    let view_data = view.storage.as_uninit_slice();

    if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
        for (dst, &val) in out_data.iter_mut().zip(view_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(val.cos());
        }
    } else {
        for (dst, idx) in out_data
            .iter_mut()
            .zip(view.layout.iter_element_indices(D::DTYPE)?)
        {
            let val = unsafe { view_data[idx].assume_init() };
            dst.write(val.cos());
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl CosKernel for f32 {
    fn cos_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(view, allocator)
    }
}

impl CosKernel for f64 {
    fn cos_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(view, allocator)
    }
}
