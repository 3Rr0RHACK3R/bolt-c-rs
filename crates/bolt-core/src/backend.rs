use crate::{
    allocator::StorageAllocator,
    device::{BackendDevice, DeviceKind},
    dtype::NativeType,
    error::Result,
    layout::Layout,
};

#[derive(Clone, Debug)]
pub struct TensorParts<S> {
    pub storage: S,
    pub layout: Layout,
}

pub trait Backend<D: NativeType>: Clone + Send + Sync + 'static {
    type Device: BackendDevice + Clone + Send + Sync + 'static;
    type Storage: Clone + Send + Sync + 'static;
    type Allocator: StorageAllocator<D, Storage = Self::Storage>;

    fn device(&self) -> &Self::Device;
    fn allocator(&self) -> Self::Allocator;
    fn device_kind(&self) -> DeviceKind {
        self.device().kind()
    }
    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize;

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()>;
    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()>;
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>>;

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;

    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;

    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;

    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<<Self as Backend<f32>>::Storage>>
    where
        Self: Backend<f32>;
}
