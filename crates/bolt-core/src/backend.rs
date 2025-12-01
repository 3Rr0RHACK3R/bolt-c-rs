use crate::{
    allocator::StorageAllocator,
    device::{BackendDevice, DeviceKind},
    dtype::{FloatType, NativeType},
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
}

pub trait CopyOp<D: NativeType>: Backend<D> {
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>>;
}

pub trait FillOp<D: NativeType>: Backend<D> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage>;
}

pub trait AddOp<D: NativeType>: Backend<D> {
    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait MulOp<D: NativeType>: Backend<D> {
    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait SubOp<D: NativeType>: Backend<D> {
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait MatmulOp<D: NativeType>: Backend<D> {
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait MeanOp<D: NativeType>: Backend<D> {
    type F32Storage: Clone + Send + Sync + 'static;

    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<Self::F32Storage>>;
}

pub trait NegOp<D: NativeType>: Backend<D> {
    fn neg(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait AbsOp<D: NativeType>: Backend<D> {
    fn abs(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait ExpOp<D: FloatType>: Backend<D> {
    fn exp(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait LogOp<D: FloatType>: Backend<D> {
    fn log(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait SqrtOp<D: FloatType>: Backend<D> {
    fn sqrt(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait SinOp<D: FloatType>: Backend<D> {
    fn sin(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait CosOp<D: FloatType>: Backend<D> {
    fn cos(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait TanhOp<D: FloatType>: Backend<D> {
    fn tanh(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait ReluOp<D: NativeType>: Backend<D> {
    fn relu(&self, layout: &Layout, storage: &Self::Storage) -> Result<TensorParts<Self::Storage>>;
}

pub trait DivOp<D: NativeType>: Backend<D> {
    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait PowOp<D: FloatType>: Backend<D> {
    fn pow(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait SumOp<D: NativeType>: Backend<D> {
    fn sum(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait ProdOp<D: NativeType>: Backend<D> {
    fn prod(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait MinOp<D: NativeType>: Backend<D> {
    fn min(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait MaxOp<D: NativeType>: Backend<D> {
    fn max(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage>>;
}

pub trait ArgminOp<D: NativeType>: Backend<D> {
    type I32Storage: Clone + Send + Sync + 'static;
    fn argmin(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::I32Storage>>;
}

pub trait ArgmaxOp<D: NativeType>: Backend<D> {
    type I32Storage: Clone + Send + Sync + 'static;
    fn argmax(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[usize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::I32Storage>>;
}
