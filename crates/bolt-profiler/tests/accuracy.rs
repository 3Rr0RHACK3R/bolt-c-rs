use bolt_core::backend::FillOp;
use bolt_core::layout::Layout;
use bolt_core::shape::ConcreteShape;
use bolt_cpu::CpuBackend;
use bolt_profiler::{ProfiledBackend, QueryBuilder, TrackingAllocator};
use serial_test::serial;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator::new();

fn make_layout(len: usize) -> Layout {
    let shape = ConcreteShape::from_slice(&[len]).expect("shape");
    Layout::contiguous(shape)
}

#[test]
#[serial]
fn allocator_tracks_fill_ops() {
    let backend = Arc::new(ProfiledBackend::new(CpuBackend::new(), None));
    let layout = make_layout(1024);

    let _ = backend.fill(&layout, 1.0f32).unwrap();

    let registry = backend.registry();
    let stats = registry.lock();

    let fill_ops: Vec<_> = QueryBuilder::new(&stats).name_contains("fill").collect();

    assert!(!fill_ops.is_empty(), "Should have recorded fill op");
    let record = fill_ops[0];
    assert_eq!(record.name, "fill");
    assert!(record.stats.last_report.is_some());

    let report = record.stats.last_report.as_ref().unwrap();
    
    // Updated for new profiler axes
    // CpuBackend uses the backend allocator (device stats)
    assert!(report.memory.device.available, "Device memory stats should be available for CpuBackend");
    assert!(report.memory.device.alloc_count > 0);
}

#[test]
#[serial]
fn scope_tracks_nested_ops() {
    let backend: Arc<ProfiledBackend<CpuBackend>> =
        Arc::new(ProfiledBackend::new(CpuBackend::new(), None));
    backend.clear_stats();

    let _scope_id = backend.begin_scope("my_scope");
    let layout = make_layout(512);
    let _ = backend.fill(&layout, 2.0f32).unwrap();
    let _ = backend.fill(&layout, 3.0f32).unwrap();
    let report = backend.end_scope().expect("scope report");

    assert!(report.time.host.wall_time.as_nanos() > 0);

    let registry = backend.registry();
    let stats = registry.lock();

    let scopes: Vec<_> = stats.top_level_ops().collect();
    assert_eq!(scopes.len(), 1, "Should have one top-level scope");
    assert_eq!(scopes[0].name, "my_scope");

    let children: Vec<_> = stats.children_of(scopes[0].id).collect();
    assert_eq!(children.len(), 2, "Scope should have 2 fill children");
}

#[test]
#[serial]
fn tracking_allocator_fallback() {
    let backend = Arc::new(
        ProfiledBackend::builder(CpuBackend::new())
            .with_tracking_allocator(&GLOBAL)
            .build(),
    );
    let layout = make_layout(256);

    let _ = backend.fill(&layout, 1.0f32).unwrap();

    let registry = backend.registry();
    let stats = registry.lock();

    let ops: Vec<_> = stats.top_level_ops().collect();
    assert!(!ops.is_empty());
}

#[test]
#[serial]
fn sample_rate_controls_profiling() {
    let backend = Arc::new(
        ProfiledBackend::builder(CpuBackend::new())
            .sample_rate(0.0)
            .build(),
    );

    let layout = make_layout(64);
    for _ in 0..10 {
        let _ = backend.fill(&layout, 1.0f32).unwrap();
    }

    let registry = backend.registry();
    let stats = registry.lock();
    assert!(
        stats.ops().is_empty(),
        "No ops should be recorded with 0% sample rate"
    );
}
