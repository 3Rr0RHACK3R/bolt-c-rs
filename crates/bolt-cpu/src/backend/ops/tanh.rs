use bolt_core::{
    Float, StorageAllocator, TensorParts,
    error::{Error, Result},
    layout::Layout,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait TanhKernel: Float {
    fn tanh_kernel(
        _view: CpuTensorView<'_, Self>,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "tanh not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn tanh<D>(
    view: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: Float,
{
    let shape = view.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;
    let view_data = view.storage.as_uninit_slice();

    if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
        for (dst, &val) in out_data.iter_mut().zip(view_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(val.tanh());
        }
    } else {
        for (dst, idx) in out_data
            .iter_mut()
            .zip(view.layout.iter_element_indices(D::DTYPE)?)
        {
            let val = unsafe { view_data[idx].assume_init() };
            dst.write(val.tanh());
        }
    }

    let layout = Layout::contiguous(shape.clone());
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl TanhKernel for f32 {
    fn tanh_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        tanh(view, allocator)
    }
}

impl TanhKernel for f64 {
    fn tanh_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        tanh(view, allocator)
    }
}
