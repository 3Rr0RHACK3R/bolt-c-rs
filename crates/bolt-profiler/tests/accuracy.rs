use bolt_core::backend::FillOp;
use bolt_core::layout::Layout;
use bolt_core::shape::Shape;
use bolt_cpu::CpuBackend;
use bolt_profiler::{HostMemTracker, ProfiledBackend};
use serial_test::serial;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: HostMemTracker = HostMemTracker::new();

fn make_layout(len: usize) -> Layout {
    let shape = Shape::from_slice(&[len]).expect("shape");
    Layout::contiguous(shape)
}

#[test]
#[serial]
fn host_mem_tracker_records_fill_ops() {
    GLOBAL.reset_counts();
    let (backend, profiler) = ProfiledBackend::wrap_with_host_mem(CpuBackend::new(), &GLOBAL);
    let layout = make_layout(1024);

    let _ = backend.fill(&layout, 1.0f32).unwrap();

    let registry = profiler.registry();
    let stats = registry.lock();

    let fill_ops: Vec<_> = stats.ops().values().filter(|r| r.name == "fill").collect();

    assert!(!fill_ops.is_empty(), "Should have recorded fill op");
    let report = fill_ops[0].stats.last_report.as_ref().unwrap();

    assert!(
        report.memory.host.available,
        "Host memory stats should be available when tracker is provided"
    );
    assert!(report.memory.host.alloc_count > 0);
    assert!(report.memory.host.bytes_granted > 0);
}

#[test]
#[serial]
fn scope_tracks_nested_ops_with_children() {
    let backend: Arc<ProfiledBackend<CpuBackend>> =
        Arc::new(ProfiledBackend::new(CpuBackend::new(), Some(&GLOBAL)));
    let profiler = backend.profiler().clone();
    profiler.clear();
    GLOBAL.reset_counts();

    profiler.with_scope("my_scope", || {
        let layout = make_layout(512);
        let _ = backend.fill(&layout, 2.0f32).unwrap();
        let _ = backend.fill(&layout, 3.0f32).unwrap();
    });

    let registry = profiler.registry();
    let stats = registry.lock();

    let scopes: Vec<_> = stats.top_level_ops().collect();
    assert_eq!(scopes.len(), 1, "Should have one top-level scope");
    assert_eq!(scopes[0].name, "my_scope");

    let children: Vec<_> = stats.children_of(scopes[0].id).collect();
    assert!(
        children.len() >= 2,
        "Scope should have recorded child fill operations"
    );
    assert!(
        children
            .iter()
            .all(|c| c.category != bolt_profiler::OpCategory::UserScope),
        "Children should be op records, not scopes"
    );
}
