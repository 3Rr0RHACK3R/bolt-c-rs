use bolt_core::{
    Layout, NativeType, StorageAllocator, TensorParts, error::Result, shape::ConcreteShape,
};

use crate::backend::{CpuStorage, allocator::CpuAllocator, storage::read_into_uninit_slice};

pub trait CopyKernel: NativeType {
    fn copy_kernel(
        storage: &CpuStorage<Self>,
        layout: &Layout,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        let shape = ConcreteShape::from_slice(layout.shape())?;
        let numel = shape.num_elements();
        let mut dst: CpuStorage<Self> = allocator.allocate(numel)?;
        {
            let slice = dst.try_as_uninit_slice_mut()?;
            read_into_uninit_slice(storage, layout, slice)?;
        }
        let layout = Layout::contiguous(shape);
        Ok(TensorParts {
            storage: dst,
            layout,
        })
    }
}

impl<T: NativeType> CopyKernel for T {}
