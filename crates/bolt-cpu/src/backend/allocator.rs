use std::{marker::PhantomData, sync::Arc};

use bolt_core::{
    allocator::StorageAllocator,
    dtype::{DType, NativeType},
    error::Result,
};

use super::context::CpuContext;

use super::storage::{CpuStorage, StorageBlock, make_cpu_handle};

#[derive(Clone, Debug)]
pub struct CpuAllocator<D: NativeType> {
    #[allow(dead_code)]
    context: Arc<CpuContext>,
    _marker: PhantomData<D>,
}

impl<D: NativeType> CpuAllocator<D> {
    pub fn new(context: Arc<CpuContext>) -> Self {
        Self {
            context,
            _marker: PhantomData,
        }
    }

    pub fn num_threads(&self) -> usize {
        1
    }
}

impl<D: NativeType> StorageAllocator<D> for CpuAllocator<D> {
    type Storage = CpuStorage<D>;

    fn allocate(&self, len: usize) -> Result<Self::Storage> {
        let handle = make_cpu_handle::<D>(len)?;
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(len, false)),
        ))
    }

    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage> {
        let handle = make_cpu_handle::<D>(len)?;
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(len, true)),
        ))
    }

    fn allocate_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let len = len_bytes / dtype.size_in_bytes();
        let handle = make_cpu_handle::<D>(len)?;
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(len, false)),
        ))
    }

    fn allocate_zeroed_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let len = len_bytes / dtype.size_in_bytes();
        let handle = make_cpu_handle::<D>(len)?;
        Ok(CpuStorage::new(
            handle,
            Arc::new(StorageBlock::new(len, true)),
        ))
    }
}
