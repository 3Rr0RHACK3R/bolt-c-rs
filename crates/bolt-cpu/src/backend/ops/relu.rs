use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait ReluKernel: NativeType {
    fn relu_kernel(
        _view: CpuTensorView<'_, Self>,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "relu not implemented for {}",
            Self::DTYPE
        )))
    }
}

pub fn relu<D>(
    view: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + PartialOrd,
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
            dst.write(if val > zero { val } else { zero });
        }
    } else {
        for (dst, idx) in out_data
            .iter_mut()
            .zip(view.layout.iter_element_indices(D::DTYPE)?)
        {
            let val = unsafe { view_data[idx].assume_init() };
            dst.write(if val > zero { val } else { zero });
        }
    }

    let layout = Layout::contiguous(shape.clone());
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl ReluKernel for f32 {
    fn relu_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(view, allocator)
    }
}

impl ReluKernel for f64 {
    fn relu_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(view, allocator)
    }
}

impl ReluKernel for i32 {
    fn relu_kernel(
        view: CpuTensorView<'_, Self>,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(view, allocator)
    }
}
