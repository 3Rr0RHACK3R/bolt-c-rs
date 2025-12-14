use std::sync::{Arc, RwLock};

use bolt_core::BaseBackend;
use bolt_core::backend::Backend;
use bolt_core::device::DeviceKind;
use bolt_core::dtype::NativeType;
use bolt_core::layout::Layout;

use crate::Float;
use crate::grad_tape::GradTape;
use crate::graph::Graph;
use crate::operations::Autodiff;
use crate::scope::{GradContext, NoGradGuard};
use crate::storage::{AutodiffAllocator, AutodiffStorage};

impl<B, D> Clone for Autodiff<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            graph: self.graph.clone(),
            grad_enabled: self.grad_enabled.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B, D> Autodiff<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(inner: B) -> Self {
        Self {
            inner: Arc::new(inner),
            graph: Arc::new(RwLock::new(None)),
            grad_enabled: Arc::new(RwLock::new(true)),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn wrap(inner: Arc<B>) -> Self {
        Self {
            inner,
            graph: Arc::new(RwLock::new(None)),
            grad_enabled: Arc::new(RwLock::new(true)),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn inner(&self) -> &Arc<B> {
        &self.inner
    }

    pub fn begin_grad(&self) -> GradContext<B, D> {
        let graph = Graph::new();
        *self.graph.write().unwrap() = Some(graph);
        GradContext::new(self.clone())
    }

    pub fn no_grad(&self) -> NoGradGuard<B, D> {
        NoGradGuard::new(self)
    }

    pub fn with_tape<F, R>(&self, f: F) -> crate::Result<R>
    where
        F: FnOnce(&mut GradTape<B, D>) -> crate::Result<R>,
    {
        let ctx = self.begin_grad();
        let mut tape = GradTape::new(&ctx);
        f(&mut tape)
    }

    pub(crate) fn graph(&self) -> &Arc<RwLock<Option<Graph<B, D>>>> {
        &self.graph
    }

    pub(crate) fn grad_enabled_lock(&self) -> &Arc<RwLock<bool>> {
        &self.grad_enabled
    }
}

impl<B, D> Backend for Autodiff<B, D>
where
    B: BaseBackend,
    D: Float,
{
    type Device = B::Device;
    type Storage<T: NativeType> = AutodiffStorage<B::Storage<T>>;
    type Allocator<T: NativeType> = AutodiffAllocator<B::Allocator<T>>;

    fn device(&self) -> &Self::Device {
        self.inner.device()
    }

    fn allocator<T: NativeType>(&self) -> Self::Allocator<T> {
        AutodiffAllocator::new(self.inner.allocator::<T>())
    }

    fn device_kind(&self) -> DeviceKind {
        self.inner.device_kind()
    }

    fn storage_len_bytes<T: NativeType>(&self, storage: &Self::Storage<T>) -> usize {
        self.inner.storage_len_bytes(&storage.inner)
    }

    fn read<T: NativeType>(
        &self,
        storage: &Self::Storage<T>,
        layout: &Layout,
        dst: &mut [T],
    ) -> bolt_core::Result<()> {
        self.inner.read(&storage.inner, layout, dst)
    }

    fn write<T: NativeType>(
        &self,
        storage: &mut Self::Storage<T>,
        layout: &Layout,
        src: &[T],
    ) -> bolt_core::Result<()> {
        self.inner.write(&mut storage.inner, layout, src)
    }
}
