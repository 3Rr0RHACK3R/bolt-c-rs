use bolt_core::Result as CoreResult;
use bolt_core::backend::{Backend, FillOp};
use bolt_core::device::{BackendDevice, DeviceKind};
use bolt_core::dtype::NativeType;
use bolt_core::layout::Layout;
use bolt_core::shape::ConcreteShape;
use bolt_core::{AllocatorDiagnostics, StorageAllocator};
use bolt_cpu::CpuBackend;
use bolt_profiler::{OpCategory, ProfiledBackend};
use bolt_tensor::Tensor;
use std::sync::Arc;

#[derive(Clone)]
struct NoDiagStorage {
    bytes: usize,
}

#[derive(Clone)]
struct NoDiagAllocator;

impl<D: NativeType> StorageAllocator<D> for NoDiagAllocator {
    type Storage = NoDiagStorage;

    fn allocate(&self, len: usize) -> CoreResult<Self::Storage> {
        Ok(NoDiagStorage {
            bytes: len * D::DTYPE.size_in_bytes(),
        })
    }

    fn allocate_zeroed(&self, len: usize) -> CoreResult<Self::Storage> {
        Ok(NoDiagStorage {
            bytes: len * D::DTYPE.size_in_bytes(),
        })
    }
}

impl AllocatorDiagnostics for NoDiagAllocator {}

#[derive(Clone, Copy)]
struct TestDevice;

impl BackendDevice for TestDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

#[derive(Clone)]
struct NoDiagBackend {
    device: TestDevice,
}

impl NoDiagBackend {
    fn new() -> Self {
        Self { device: TestDevice }
    }
}

impl Backend for NoDiagBackend {
    type Device = TestDevice;
    type Storage<D: NativeType> = NoDiagStorage;
    type Allocator<D: NativeType> = NoDiagAllocator;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocator<D: NativeType>(&self) -> Self::Allocator<D> {
        NoDiagAllocator
    }

    fn storage_len_bytes<D: NativeType>(&self, storage: &Self::Storage<D>) -> usize {
        storage.bytes
    }

    fn read<D: NativeType>(
        &self,
        _storage: &Self::Storage<D>,
        _layout: &Layout,
        _dst: &mut [D],
    ) -> CoreResult<()> {
        Ok(())
    }

    fn write<D: NativeType>(
        &self,
        _storage: &mut Self::Storage<D>,
        _layout: &Layout,
        _src: &[D],
    ) -> CoreResult<()> {
        Ok(())
    }
}

impl FillOp<f32> for NoDiagBackend {
    fn fill(&self, layout: &Layout, value: f32) -> CoreResult<Self::Storage<f32>> {
        let len = layout.num_elements();
        let mut data = Vec::with_capacity(len);
        data.resize(len, value);
        std::hint::black_box(&data);
        <NoDiagAllocator as StorageAllocator<f32>>::allocate(&self.allocator::<f32>(), len)
    }
}

fn make_layout(len: usize) -> Layout {
    let shape = ConcreteShape::from_slice(&[len]).expect("shape");
    Layout::contiguous(shape)
}

#[test]
fn backend_allocator_diagnostics_used() -> CoreResult<()> {
    let (backend, profiler) = ProfiledBackend::wrap(CpuBackend::new());
    let layout = make_layout(8);
    let _ = backend.fill(&layout, 1.0f32)?;

    let registry = profiler.registry();
    let stats = registry.lock();

    let fills: Vec<_> = stats.ops().values().filter(|r| r.name == "fill").collect();

    assert!(!fills.is_empty(), "Should have fill op");
    let report = fills[0].stats.last_report.as_ref().expect("fill stats");

    assert!(
        report.memory.device.available,
        "Backend allocator diagnostics should be used"
    );
    assert!(report.memory.device.alloc_count > 0);
    assert!(report.memory.device.bytes_granted > 0);
    Ok(())
}

#[test]
fn diagnostics_unavailable_without_backend_or_fallback() -> CoreResult<()> {
    let (backend, profiler) = ProfiledBackend::wrap(NoDiagBackend::new());
    let layout = make_layout(2);
    let _ = backend.fill(&layout, 3.0f32)?;

    let registry = profiler.registry();
    let stats = registry.lock();

    let fills: Vec<_> = stats.ops().values().filter(|r| r.name == "fill").collect();

    assert!(!fills.is_empty());
    let report = fills[0].stats.last_report.as_ref().expect("fill stats");

    assert!(
        !report.memory.device.available,
        "Diagnostics should be unavailable"
    );
    assert_eq!(report.memory.device.bytes_granted, 0);
    assert_eq!(report.memory.device.bytes_requested, 0);
    Ok(())
}

#[test]
fn scopes_record_parent_child_relationships() -> CoreResult<()> {
    let backend: Arc<ProfiledBackend<CpuBackend>> =
        Arc::new(ProfiledBackend::new(CpuBackend::new(), None));
    let profiler = backend.profiler().clone();
    profiler.clear();

    profiler.with_scope("tensor_add_scope", || {
        let shape = [1024, 1024];
        let a = Tensor::<_, f32>::zeros(&backend, &shape).unwrap();
        let b = Tensor::<_, f32>::zeros(&backend, &shape).unwrap();
        let c = a.add(&b).unwrap();
        drop(c);
    });

    let registry = profiler.registry();
    let stats = registry.lock();

    let top_level: Vec<_> = stats.top_level_ops().collect();
    assert_eq!(top_level.len(), 1, "Should have one top-level scope");
    assert_eq!(top_level[0].name, "tensor_add_scope");
    assert_eq!(top_level[0].category, OpCategory::UserScope);

    let children: Vec<_> = stats.children_of(top_level[0].id).collect();
    assert!(
        children.len() >= 1,
        "Scope should have child ops recorded within it"
    );

    Ok(())
}
