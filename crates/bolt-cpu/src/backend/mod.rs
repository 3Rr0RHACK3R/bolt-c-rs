mod allocator;
mod context;
mod memory_pool;
pub mod ops;
mod storage;

use std::sync::Arc;

use bolt_core::{
    TensorParts, TensorView,
    allocator::StorageAllocator,
    backend::{
        AbsOp, AddOp, ArgmaxOp, ArgminOp, Backend, BroadcastToOp, CopyOp, CosOp, DivOp, ExpOp,
        FillOp, LogOp, MatmulOp, MaxOp, MeanOp, MinOp, MulOp, NegOp, PowOp, ProdOp, ReluOp,
        ReshapeOp, SinOp, SqrtOp, SqueezeOp, SubOp, SumOp, TanhOp, TransposeOp, UnsqueezeOp,
    },
    device::{BackendDevice, DeviceKind},
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
};

pub use storage::{CpuStorage, CpuTensorView};

#[cfg(feature = "diagnostics")]
use allocator::CpuAllocTelemetry;
use allocator::CpuAllocator;
use context::CpuContext;
use memory_pool::MemoryPool;
use ops::{
    AbsKernel, AddKernel, ArgmaxKernel, ArgminKernel, CopyKernel, CosKernel, CpuScalar, DivKernel,
    ExpKernel, LogKernel, MatmulKernel, MaxKernel, MeanKernel, MinKernel, MulKernel, NegKernel,
    PowKernel, ProdKernel, ReluKernel, SinKernel, SqrtKernel, SubKernel, SumKernel, TanhKernel,
};
use storage::{fill_storage, read_into_slice, write_from_slice};

#[derive(Clone)]
pub struct CpuBackend {
    device: Arc<CpuDevice>,
    context: Arc<CpuContext>,
    pool: Option<Arc<MemoryPool>>,
    #[cfg(feature = "diagnostics")]
    diagnostics: Arc<CpuAllocTelemetry>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Arc::new(CpuDevice),
            context: CpuContext::new(),
            pool: None,
            #[cfg(feature = "diagnostics")]
            diagnostics: Arc::new(CpuAllocTelemetry::default()),
        }
    }

    pub fn with_pooling() -> Self {
        Self {
            device: Arc::new(CpuDevice),
            context: CpuContext::new(),
            pool: Some(Arc::new(MemoryPool::new())),
            #[cfg(feature = "diagnostics")]
            diagnostics: Arc::new(CpuAllocTelemetry::default()),
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

impl Backend for CpuBackend {
    type Device = CpuDevice;
    type Storage<D: NativeType> = CpuStorage<D>;
    type Allocator<D: NativeType> = CpuAllocator<D>;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocator<D: NativeType>(&self) -> Self::Allocator<D> {
        #[cfg(feature = "diagnostics")]
        {
            CpuAllocator::with_diagnostics(
                self.context.clone(),
                Arc::clone(&self.diagnostics),
                self.pool.clone(),
            )
        }

        #[cfg(not(feature = "diagnostics"))]
        {
            if let Some(pool) = &self.pool {
                CpuAllocator::new_caching(self.context.clone(), pool.clone())
            } else {
                CpuAllocator::new(self.context.clone())
            }
        }
    }

    fn storage_len_bytes<D: NativeType>(&self, storage: &Self::Storage<D>) -> usize {
        storage.len_bytes()
    }

    fn read<D: NativeType>(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        dst: &mut [D],
    ) -> Result<()> {
        unsafe { read_into_slice(storage, layout, dst) }
    }

    fn write<D: NativeType>(
        &self,
        storage: &mut Self::Storage<D>,
        layout: &Layout,
        src: &[D],
    ) -> Result<()> {
        write_from_slice(storage, layout, src)
    }
}

impl<D> CopyOp<D> for CpuBackend
where
    D: CpuScalar + CopyKernel,
{
    fn copy(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as CopyKernel>::copy_kernel(storage, layout, &self.allocator::<D>())
    }
}

impl<D> FillOp<D> for CpuBackend
where
    D: CpuScalar,
{
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage<D>> {
        let len_bytes =
            layout
                .max_offset_bytes(D::DTYPE)?
                .checked_add(1)
                .ok_or(Error::TensorTooLarge {
                    limit: isize::MAX as usize,
                    requested: usize::MAX,
                })?;
        let mut storage = self.allocator::<D>().allocate_bytes(len_bytes, D::DTYPE)?;
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
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as AddKernel>::add_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> SubOp<D> for CpuBackend
where
    D: CpuScalar + SubKernel,
{
    fn sub(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as SubKernel>::sub_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> MatmulOp<D> for CpuBackend
where
    D: CpuScalar + MatmulKernel,
{
    fn matmul(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as MatmulKernel>::matmul_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> MulOp<D> for CpuBackend
where
    D: CpuScalar + MulKernel,
{
    fn mul(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as MulKernel>::mul_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> MeanOp<D> for CpuBackend
where
    D: CpuScalar + MeanKernel,
{
    fn mean(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as MeanKernel>::mean_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &self.allocator::<D>(),
        )
    }
}

impl<D> NegOp<D> for CpuBackend
where
    D: CpuScalar + NegKernel,
{
    fn neg(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as NegKernel>::neg_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> AbsOp<D> for CpuBackend
where
    D: CpuScalar + AbsKernel,
{
    fn abs(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as AbsKernel>::abs_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> ExpOp<D> for CpuBackend
where
    D: CpuScalar + ExpKernel,
{
    fn exp(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as ExpKernel>::exp_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> LogOp<D> for CpuBackend
where
    D: CpuScalar + LogKernel,
{
    fn log(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as LogKernel>::log_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> SqrtOp<D> for CpuBackend
where
    D: CpuScalar + SqrtKernel,
{
    fn sqrt(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as SqrtKernel>::sqrt_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> SinOp<D> for CpuBackend
where
    D: CpuScalar + SinKernel,
{
    fn sin(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as SinKernel>::sin_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> CosOp<D> for CpuBackend
where
    D: CpuScalar + CosKernel,
{
    fn cos(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as CosKernel>::cos_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> TanhOp<D> for CpuBackend
where
    D: CpuScalar + TanhKernel,
{
    fn tanh(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as TanhKernel>::tanh_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> ReluOp<D> for CpuBackend
where
    D: CpuScalar + ReluKernel,
{
    fn relu(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as ReluKernel>::relu_kernel(TensorView::new(storage, layout), &self.allocator::<D>())
    }
}

impl<D> DivOp<D> for CpuBackend
where
    D: CpuScalar + DivKernel,
{
    fn div(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as DivKernel>::div_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> PowOp<D> for CpuBackend
where
    D: CpuScalar + PowKernel,
{
    fn pow(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as PowKernel>::pow_kernel(
            TensorView::new(lhs, lhs_layout),
            TensorView::new(rhs, rhs_layout),
            &self.allocator::<D>(),
        )
    }
}

impl<D> SumOp<D> for CpuBackend
where
    D: CpuScalar + SumKernel,
{
    fn sum(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as SumKernel>::sum_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &self.allocator::<D>(),
        )
    }
}

impl<D> ProdOp<D> for CpuBackend
where
    D: CpuScalar + ProdKernel,
{
    fn prod(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as ProdKernel>::prod_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &self.allocator::<D>(),
        )
    }
}

impl<D> MinOp<D> for CpuBackend
where
    D: CpuScalar + MinKernel,
{
    fn min(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as MinKernel>::min_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &self.allocator::<D>(),
        )
    }
}

impl<D> MaxOp<D> for CpuBackend
where
    D: CpuScalar + MaxKernel,
{
    fn max(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        <D as MaxKernel>::max_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &self.allocator::<D>(),
        )
    }
}

impl<D> ArgminOp<D> for CpuBackend
where
    D: CpuScalar + ArgminKernel,
{
    fn argmin(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<i32>>> {
        let alloc_i32 = self.allocator::<i32>();
        <D as ArgminKernel>::argmin_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &alloc_i32,
        )
    }
}

impl<D> ArgmaxOp<D> for CpuBackend
where
    D: CpuScalar + ArgmaxKernel,
{
    fn argmax(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<i32>>> {
        let alloc_i32 = self.allocator::<i32>();
        <D as ArgmaxKernel>::argmax_kernel(
            TensorView::new(storage, layout),
            axes,
            keepdims,
            &alloc_i32,
        )
    }
}

impl<D> ReshapeOp<D> for CpuBackend where D: CpuScalar {}
impl<D> SqueezeOp<D> for CpuBackend where D: CpuScalar {}
impl<D> UnsqueezeOp<D> for CpuBackend where D: CpuScalar {}
impl<D> TransposeOp<D> for CpuBackend where D: CpuScalar {}
impl<D> BroadcastToOp<D> for CpuBackend where D: CpuScalar {}
