use bolt_core::{error::Result, Layout, NativeType, StorageAllocator, TensorParts};

use crate::backend::{allocator::CpuAllocator, storage::read_into_uninit_slice, CpuStorage};

pub trait CopyKernel: NativeType {
    fn copy_kernel(
        storage: &CpuStorage<Self>,
        layout: &Layout,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        let shape = layout.shape().clone();
        let numel = shape.num_elements();
        let mut dst: CpuStorage<Self> = allocator.allocate(numel)?;
        {
            let slice = dst.try_as_uninit_slice_mut()?;
            unsafe { read_into_uninit_slice(storage, layout, slice)? };
        }
        let layout = Layout::contiguous(shape);
        Ok(TensorParts {
            storage: dst,
            layout,
        })
    }
}

impl<T: NativeType> CopyKernel for T {}
