use bolt_core::Result as CoreResult;
use bolt_core::backend::{Backend, FillOp};
use bolt_core::device::{BackendDevice, DeviceKind};
use bolt_core::layout::Layout;
use bolt_core::shape::ConcreteShape;
use bolt_core::{AllocatorDiagnostics, NativeType, StorageAllocator, Tensor};
use bolt_cpu::CpuBackend;
use bolt_profiler::{OpCategory, ProfiledBackend, QueryBuilder};
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
    allocator: NoDiagAllocator,
}

impl NoDiagBackend {
    fn new() -> Self {
        Self {
            device: TestDevice,
            allocator: NoDiagAllocator,
        }
    }
}

impl Backend<f32> for NoDiagBackend {
    type Device = TestDevice;
    type Storage = NoDiagStorage;
    type Allocator = NoDiagAllocator;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocator(&self) -> Self::Allocator {
        self.allocator.clone()
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        storage.bytes
    }

    fn read(&self, _storage: &Self::Storage, _layout: &Layout, _dst: &mut [f32]) -> CoreResult<()> {
        Ok(())
    }

    fn write(
        &self,
        _storage: &mut Self::Storage,
        _layout: &Layout,
        _src: &[f32],
    ) -> CoreResult<()> {
        Ok(())
    }
}

impl FillOp<f32> for NoDiagBackend {
    fn fill(&self, layout: &Layout, value: f32) -> CoreResult<Self::Storage> {
        let len = layout.num_elements();
        let mut data = Vec::with_capacity(len);
        data.resize(len, value);
        std::hint::black_box(&data);
        <NoDiagAllocator as StorageAllocator<f32>>::allocate(&self.allocator, len)
    }
}

fn make_layout(len: usize) -> Layout {
    let shape = ConcreteShape::from_slice(&[len]).expect("shape");
    Layout::contiguous(shape)
}

#[test]
fn backend_allocator_diagnostics_used() -> CoreResult<()> {
    let backend = ProfiledBackend::new(CpuBackend::new(), None);
    let layout = make_layout(8);
    let _ = backend.fill(&layout, 1.0f32)?;

    let registry = backend.registry();
    let stats = registry.lock();

    let fills: Vec<_> = QueryBuilder::new(&stats).name_contains("fill").collect();

    assert!(!fills.is_empty(), "Should have fill op");
    let report = fills[0].stats.last_report.as_ref().expect("fill stats");

    assert!(report.memory.device.available, "Backend allocator diagnostics should be used");
    assert!(report.memory.device.alloc_count > 0);
    assert!(report.memory.device.bytes_granted > 0);
    Ok(())
}

#[test]
fn diagnostics_unavailable_without_backend_or_fallback() -> CoreResult<()> {
    let backend = ProfiledBackend::new(NoDiagBackend::new(), None);
    let layout = make_layout(2);
    let _ = backend.fill(&layout, 3.0f32)?;

    let registry = backend.registry();
    let stats = registry.lock();

    let fills: Vec<_> = QueryBuilder::new(&stats).name_contains("fill").collect();

    assert!(!fills.is_empty());
    let report = fills[0].stats.last_report.as_ref().expect("fill stats");

    assert!(!report.memory.device.available, "Diagnostics should be unavailable");
    assert_eq!(report.memory.device.bytes_granted, 0);
    assert_eq!(report.memory.device.bytes_requested, 0);
    Ok(())
}

#[test]
fn tensor_add_reports_balanced_allocs_and_deallocs() -> CoreResult<()> {
    let backend: Arc<ProfiledBackend<CpuBackend>> =
        Arc::new(ProfiledBackend::new(CpuBackend::new(), None));
    backend.clear_stats();
    let shape = [1024, 1024];

    backend.begin_scope("tensor_add_scope");

    let a = Tensor::<_, f32>::zeros(&backend, &shape)?;
    let b = Tensor::<_, f32>::zeros(&backend, &shape)?;
    let c = a.add(&b)?;
    drop(c);
    drop(b);
    drop(a);

    let report = backend.end_scope().expect("scope report");

    bolt_profiler::print_report(backend.registry());

    assert!(report.memory.device.available);
    assert_eq!(
        report.memory.device.alloc_count, report.memory.device.dealloc_count,
        "Allocs and deallocs should be balanced"
    );
    assert!(report.memory.device.alloc_count > 0);

    let registry = backend.registry();
    let stats = registry.lock();

    let top_level: Vec<_> = stats.top_level_ops().collect();
    assert_eq!(top_level.len(), 1, "Should have one top-level scope");
    assert_eq!(top_level[0].name, "tensor_add_scope");
    assert_eq!(top_level[0].category, OpCategory::UserScope);

    let children: Vec<_> = stats.children_of(top_level[0].id).collect();
    assert!(children.len() >= 1, "Scope should have child ops");

    Ok(())
}

#[test]
fn query_api_filters_correctly() -> CoreResult<()> {
    let backend = Arc::new(ProfiledBackend::new(CpuBackend::new(), None));
    backend.clear_stats();

    let small_layout = make_layout(8);
    let _ = backend.fill(&small_layout, 1.0f32)?;
    let _ = backend.fill(&small_layout, 2.0f32)?;

    let registry = backend.registry();
    let stats = registry.lock();

    let all_ops: Vec<_> = QueryBuilder::new(&stats).collect();
    assert_eq!(all_ops.len(), 2);

    let memory_ops: Vec<_> = QueryBuilder::new(&stats)
        .category(OpCategory::Memory)
        .collect();
    assert_eq!(memory_ops.len(), 2);

    let compute_ops: Vec<_> = QueryBuilder::new(&stats)
        .category(OpCategory::Compute)
        .collect();
    assert_eq!(compute_ops.len(), 0);

    Ok(())
}
