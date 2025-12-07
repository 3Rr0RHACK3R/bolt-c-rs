use bolt_core::backend::{
    AddOp, Backend, CopyOp, FillOp, MatmulOp, MeanOp, SubOp, TensorParts, TransposeOp,
};
use bolt_core::dtype::{FloatType, NativeType};
use bolt_core::error::Result;
use bolt_core::layout::Layout;

use crate::host_mem::HostMemTracker;
use crate::profiler::Profiler;
use crate::registry::OpCategory;

fn shapes_from_layout(layout: &Layout) -> Vec<usize> {
    layout.shape().to_vec()
}

fn profile_op<D, B, F, R>(
    backend: &ProfiledBackend<B>,
    name: &'static str,
    category: OpCategory,
    shapes: Vec<Vec<usize>>,
    f: F,
) -> R
where
    D: NativeType,
    B: Backend<D>,
    F: FnOnce(&B) -> R,
{
    let allocator = backend.inner.allocator();
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

impl<D: NativeType, B: Backend<D>> Backend<D> for ProfiledBackend<B> {
    type Device = B::Device;
    type Storage = B::Storage;
    type Allocator = B::Allocator;

    fn device(&self) -> &Self::Device {
        self.inner.device()
    }

    fn allocator(&self) -> Self::Allocator {
        self.inner.allocator()
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        self.inner.storage_len_bytes(storage)
    }

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()> {
        profile_op(
            self,
            "read",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.read(storage, layout, dst),
        )
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        profile_op(
            self,
            "write",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.write(storage, layout, src),
        )
    }
}

impl<D: NativeType, B: CopyOp<D> + Backend<D>> CopyOp<D> for ProfiledBackend<B> {
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        profile_op(
            self,
            "copy",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.copy(storage, layout),
        )
    }
}

impl<D: NativeType, B: FillOp<D> + Backend<D>> FillOp<D> for ProfiledBackend<B> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage> {
        profile_op(
            self,
            "fill",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.fill(layout, value),
        )
    }
}

impl<D: NativeType, B: AddOp<D> + Backend<D>> AddOp<D> for ProfiledBackend<B> {
    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
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

impl<D: NativeType, B: SubOp<D> + Backend<D>> SubOp<D> for ProfiledBackend<B> {
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
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

impl<D: NativeType, B: MatmulOp<D> + Backend<D>> MatmulOp<D> for ProfiledBackend<B> {
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
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

impl<D: FloatType, B: MeanOp<D> + Backend<D>> MeanOp<D> for ProfiledBackend<B> {
    fn mean(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage>> {
        profile_op(
            self,
            "mean",
            OpCategory::Compute,
            vec![shapes_from_layout(layout)],
            |inner| inner.mean(layout, storage, axes, keepdims),
        )
    }
}

impl<D: NativeType, B: TransposeOp<D> + Backend<D>> TransposeOp<D> for ProfiledBackend<B> {
    fn transpose(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        axis_a: isize,
        axis_b: isize,
    ) -> Result<TensorParts<Self::Storage>> {
        profile_op(
            self,
            "transpose",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            |inner| inner.transpose(storage, layout, axis_a, axis_b),
        )
    }
}
