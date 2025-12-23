use bolt_core::backend::{
    AddOp, Backend, CopyOp, FillOp, MatmulOp, MeanOp, NegOp, ReshapeOp, SubOp, SumOp, TensorParts,
    TransposeOp,
};
use bolt_core::dtype::{Float, NativeType};
use bolt_core::error::Result;
use bolt_core::layout::Layout;

use crate::host_mem::HostMemTracker;
use crate::profiler::Profiler;
use crate::registry::OpCategory;

fn shapes_from_layout(layout: &Layout) -> Vec<usize> {
    layout.shape().to_vec()
}

fn profile_op<B, F, R>(
    backend: &ProfiledBackend<B>,
    name: &'static str,
    category: OpCategory,
    shapes: Vec<Vec<usize>>,
    f: F,
) -> R
where
    B: Backend,
    F: FnOnce(&B) -> R,
{
    let allocator: B::Allocator<f32> = backend.inner.allocator();
    let active = backend.profiler.begin_op(&allocator);
    let result = f(&backend.inner);
    backend
        .profiler
        .end_op(active, &allocator, name, category, shapes);
    result
}

#[derive(Debug)]
pub struct ProfiledBackend<B> {
    inner: B,
    profiler: Profiler,
}

impl<B: Clone> Clone for ProfiledBackend<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            profiler: self.profiler.clone(),
        }
    }
}

impl<B> ProfiledBackend<B> {
    pub fn new(inner: B, host_mem: Option<&'static HostMemTracker>) -> Self {
        let profiler = Profiler::new(host_mem);
        Self { inner, profiler }
    }

    pub fn wrap(inner: B) -> (Self, Profiler) {
        let profiler = Profiler::new(None);
        let backend = Self {
            inner,
            profiler: profiler.clone(),
        };
        (backend, profiler)
    }

    pub fn wrap_with_host_mem(inner: B, host_mem: &'static HostMemTracker) -> (Self, Profiler) {
        let profiler = Profiler::new(Some(host_mem));
        let backend = Self {
            inner,
            profiler: profiler.clone(),
        };
        (backend, profiler)
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    pub fn registry(&self) -> std::sync::Arc<parking_lot::Mutex<crate::registry::Registry>> {
        self.profiler.registry()
    }

    pub fn clear_stats(&self) {
        self.profiler.clear();
    }
}

impl<B: Backend> Backend for ProfiledBackend<B> {
    type Device = B::Device;
    type Storage<D: NativeType> = B::Storage<D>;
    type Allocator<D: NativeType> = B::Allocator<D>;

    fn device(&self) -> &Self::Device {
        self.inner.device()
    }

    fn allocator<D: NativeType>(&self) -> Self::Allocator<D> {
        self.inner.allocator()
    }

    fn storage_len_bytes<D: NativeType>(&self, storage: &Self::Storage<D>) -> usize {
        self.inner.storage_len_bytes(storage)
    }

    fn read<D: NativeType>(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        dst: &mut [D],
    ) -> Result<()> {
        profile_op(
            self,
            "read",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.read(storage, layout, dst),
        )
    }

    fn write<D: NativeType>(
        &self,
        storage: &mut Self::Storage<D>,
        layout: &Layout,
        src: &[D],
    ) -> Result<()> {
        profile_op(
            self,
            "write",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.write(storage, layout, src),
        )
    }
}

impl<D: NativeType, B: CopyOp<D> + Backend> CopyOp<D> for ProfiledBackend<B> {
    fn copy(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "copy",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.copy(storage, layout),
        )
    }
}

impl<D: NativeType, B: FillOp<D> + Backend> FillOp<D> for ProfiledBackend<B> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage<D>> {
        profile_op(
            self,
            "fill",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.fill(layout, value),
        )
    }
}

impl<D: NativeType, B: AddOp<D> + Backend> AddOp<D> for ProfiledBackend<B> {
    fn add(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "add",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout),
            ],
            |inner| inner.add(lhs, rhs, lhs_layout, rhs_layout),
        )
    }
}

impl<D: NativeType, B: SubOp<D> + Backend> SubOp<D> for ProfiledBackend<B> {
    fn sub(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "sub",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout),
            ],
            |inner| inner.sub(lhs, rhs, lhs_layout, rhs_layout),
        )
    }
}

impl<D: NativeType, B: NegOp<D> + Backend> NegOp<D> for ProfiledBackend<B> {
    fn neg(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "neg",
            OpCategory::Compute,
            vec![shapes_from_layout(layout)],
            |inner| inner.neg(layout, storage),
        )
    }
}

impl<D: NativeType, B: MatmulOp<D> + Backend> MatmulOp<D> for ProfiledBackend<B> {
    fn matmul(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "matmul",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout),
            ],
            |inner| inner.matmul(lhs, rhs, lhs_layout, rhs_layout),
        )
    }
}

impl<D: Float, B: MeanOp<D> + Backend> MeanOp<D> for ProfiledBackend<B> {
    fn mean(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "mean",
            OpCategory::Compute,
            vec![shapes_from_layout(layout)],
            |inner| inner.mean(layout, storage, axes, keepdims),
        )
    }
}

impl<D: NativeType, B: TransposeOp<D> + Backend> TransposeOp<D> for ProfiledBackend<B> {
    fn transpose(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        axis_a: isize,
        axis_b: isize,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "transpose",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.transpose(storage, layout, axis_a, axis_b),
        )
    }
}

impl<D: NativeType, B: Backend> ReshapeOp<D> for ProfiledBackend<B> {}

impl<D: NativeType, B: SumOp<D> + Backend> SumOp<D> for ProfiledBackend<B> {
    fn sum(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        profile_op(
            self,
            "sum",
            OpCategory::Compute,
            vec![shapes_from_layout(layout)],
            |inner| inner.sum(layout, storage, axes, keepdims),
        )
    }
}
