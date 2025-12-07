use std::sync::{Arc, RwLock};

use bolt_core::backend::Backend;
use bolt_core::device::DeviceKind;
use bolt_core::layout::Layout;

use crate::Float;
use crate::device::AutodiffDevice;
use crate::graph::Graph;
use crate::operations::Autodiff;
use crate::scope::{GradContext, NoGradGuard};
use crate::storage::{AutodiffAllocator, AutodiffStorage};

impl<B, D> Clone for Autodiff<B, D>
where
    B: Backend<D>,
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
    B: Backend<D>,
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

    pub(crate) fn graph(&self) -> &Arc<RwLock<Option<Graph<B, D>>>> {
        &self.graph
    }

    pub(crate) fn grad_enabled_lock(&self) -> &Arc<RwLock<bool>> {
        &self.grad_enabled
    }
}

impl<B, D> Backend<D> for Autodiff<B, D>
where
    B: Backend<D>,
    D: Float,
{
    type Device = AutodiffDevice<B::Device>;
    type Storage = AutodiffStorage<B::Storage>;
    type Allocator = AutodiffAllocator<B::Allocator>;

    fn device(&self) -> &Self::Device {
        unsafe { &*(self.inner.device() as *const B::Device as *const AutodiffDevice<B::Device>) }
    }

    fn allocator(&self) -> Self::Allocator {
        AutodiffAllocator::new(self.inner.allocator())
    }

    fn device_kind(&self) -> DeviceKind {
        self.inner.device_kind()
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        self.inner.storage_len_bytes(&storage.inner)
    }

    fn read(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        dst: &mut [D],
    ) -> bolt_core::Result<()> {
        self.inner.read(&storage.inner, layout, dst)
    }

    fn write(
        &self,
        storage: &mut Self::Storage,
        layout: &Layout,
        src: &[D],
    ) -> bolt_core::Result<()> {
        self.inner.write(&mut storage.inner, layout, src)
    }
}
