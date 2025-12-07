use bolt_core::allocator::StorageAllocator;
use bolt_core::dtype::NativeType;

use crate::handle::Handle;

#[derive(Clone)]
pub struct AutodiffStorage<S> {
    pub(crate) inner: S,
    pub(crate) handle: Handle,
    pub(crate) requires_grad: bool,
}

impl<S> AutodiffStorage<S> {
    pub fn new(inner: S, handle: Handle, requires_grad: bool) -> Self {
        Self {
            inner,
            handle,
            requires_grad,
        }
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

#[derive(Clone)]
pub struct AutodiffAllocator<A> {
    inner: A,
}

impl<A> AutodiffAllocator<A> {
    pub fn new(inner: A) -> Self {
        Self { inner }
    }
}

impl<A: bolt_core::allocator::AllocatorDiagnostics> bolt_core::allocator::AllocatorDiagnostics
    for AutodiffAllocator<A>
{
    fn capabilities(&self) -> bolt_core::allocator::DiagnosticsCaps {
        self.inner.capabilities()
    }

    fn snapshot(&self) -> bolt_core::allocator::AllocatorSnapshot {
        self.inner.snapshot()
    }

    fn begin_scope(&self) {
        self.inner.begin_scope()
    }

    fn end_scope(&self) -> Option<bolt_core::allocator::AllocatorSnapshot> {
        self.inner.end_scope()
    }
}

impl<D, A> StorageAllocator<D> for AutodiffAllocator<A>
where
    D: NativeType,
    A: StorageAllocator<D>,
{
    type Storage = AutodiffStorage<A::Storage>;

    fn allocate(&self, num_elements: usize) -> bolt_core::error::Result<Self::Storage> {
        let inner = self.inner.allocate(num_elements)?;
        Ok(AutodiffStorage::new(inner, Handle::NONE, false))
    }

    fn allocate_zeroed(&self, num_elements: usize) -> bolt_core::error::Result<Self::Storage> {
        let inner = self.inner.allocate_zeroed(num_elements)?;
        Ok(AutodiffStorage::new(inner, Handle::NONE, false))
    }
}
