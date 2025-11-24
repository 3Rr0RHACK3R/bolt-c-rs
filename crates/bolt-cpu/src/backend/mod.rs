mod allocator;
mod context;
mod layout_utils;
mod ops;
mod storage;

use std::sync::Arc;

use bolt_core::{
    TensorParts, TensorView,
    allocator::StorageAllocator,
    backend::{AddOp, Backend, CopyOp, FillOp, MatmulOp, MeanOp, SubOp},
    device::{BackendDevice, DeviceKind},
    error::{Error, Result},
    layout::Layout,
};

pub use storage::CpuStorage;

use allocator::CpuAllocator;
use context::CpuContext;
use ops::{AddKernel, CopyKernel, CpuScalar, MatmulKernel, MeanKernel, SubKernel};
use storage::{fill_storage, read_into_slice, write_from_slice};

#[derive(Clone)]
pub struct CpuBackend {
    device: Arc<CpuDevice>,
    context: Arc<CpuContext>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Arc::new(CpuDevice),
            context: CpuContext::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CpuDevice;

impl BackendDevice for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

impl<D> Backend<D> for CpuBackend
where
    D: CpuScalar,
{
    type Device = CpuDevice;
    type Storage = CpuStorage<D>;
    type Allocator = CpuAllocator<D>;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocator(&self) -> Self::Allocator {
        CpuAllocator::new(self.context.clone())
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        storage.len_bytes()
    }

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()> {
        unsafe { read_into_slice(storage, layout, dst) }
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        write_from_slice(storage, layout, src)
    }
}

impl<D> CopyOp<D> for CpuBackend
where
    D: CpuScalar + CopyKernel,
{
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        <D as CopyKernel>::copy_kernel(storage, layout, &self.allocator())
    }
}

impl<D> FillOp<D> for CpuBackend
where
    D: CpuScalar,
{
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage> {
        let len_bytes = layout
            .max_offset_bytes(D::DTYPE)?
            .checked_add(1)
            .ok_or_else(|| Error::TensorTooLarge {
                limit: isize::MAX as usize,
                requested: usize::MAX,
            })?;
        let mut storage = self.allocator().allocate_bytes(len_bytes, D::DTYPE)?;
        fill_storage(&mut storage, layout, value)?;
        Ok(storage)
    }
}

impl<D> AddOp<D> for CpuBackend
where
    D: CpuScalar + AddKernel,
{
    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        <D as AddKernel>::add_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator(),
        )
    }
}

impl<D> SubOp<D> for CpuBackend
where
    D: CpuScalar + SubKernel,
{
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        <D as SubKernel>::sub_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator(),
        )
    }
}

impl<D> MatmulOp<D> for CpuBackend
where
    D: CpuScalar + MatmulKernel,
{
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        <D as MatmulKernel>::matmul_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator(),
        )
    }
}

impl<D> MeanOp<D> for CpuBackend
where
    D: CpuScalar + MeanKernel,
{
    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<<Self as Backend<f32>>::Storage>>
    where
        Self: Backend<f32>,
    {
        <D as MeanKernel>::mean_f32_kernel(
            TensorView::new(storage, layout),
            &<Self as Backend<f32>>::allocator(self),
        )
    }
}
