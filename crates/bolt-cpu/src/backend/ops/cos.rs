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
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "cos not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub fn cos<D>(
    input: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: FloatType + num_traits::Float,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(val.cos());
        }
    } else {
        let iter = input.layout.iter_offsets(D::DTYPE)?;
        for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
            debug_assert_eq!(idx_bytes % elem_size, 0);
            let idx = idx_bytes / elem_size;
            let val = unsafe { input_data[idx].assume_init() };
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
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(input, alloc)
    }
}

impl CosKernel for f64 {
    fn cos_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(input, alloc)
    }
}
