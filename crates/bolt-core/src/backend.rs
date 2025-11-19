use std::marker::PhantomData;

use crate::{
    allocator::StorageAllocator, device::BackendDevice, dtype::NativeType, error::Result,
    layout::Layout,
};

pub struct KernelContext<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    backend: &'a B,
    _marker: PhantomData<D>,
}

impl<'a, B, D> KernelContext<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    pub fn new(backend: &'a B) -> Self {
        Self {
            backend,
            _marker: PhantomData,
        }
    }

    pub fn backend(&self) -> &'a B {
        self.backend
    }

    pub fn allocator(&self) -> &'a B::Allocator {
        self.backend.allocator()
    }
}

pub trait Backend<D: NativeType>: Clone + Send + Sync + 'static {
    type Device: BackendDevice + Clone + Send + Sync + 'static;
    type Storage: Clone + Send + Sync + 'static;
    type Allocator: StorageAllocator<D, Storage = Self::Storage>;

    fn device(&self) -> &Self::Device;
    fn allocator(&self) -> &Self::Allocator;

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()>;
    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()>;
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<(Self::Storage, Layout)>;

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Layout)>;

    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Layout)>;

    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<(Self::Storage, Layout)>;
}
