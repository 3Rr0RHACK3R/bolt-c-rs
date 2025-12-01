use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait ReluKernel: NativeType {
    fn relu_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "relu not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub fn relu<D>(
    input: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + PartialOrd,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();
    let zero = D::default();

    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(if val > zero { val } else { zero });
        }
    } else {
        let iter = input.layout.iter_offsets(D::DTYPE)?;
        for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
            debug_assert_eq!(idx_bytes % elem_size, 0);
            let idx = idx_bytes / elem_size;
            let val = unsafe { input_data[idx].assume_init() };
            dst.write(if val > zero { val } else { zero });
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl ReluKernel for f32 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}

impl ReluKernel for f64 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}

impl ReluKernel for i32 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}
