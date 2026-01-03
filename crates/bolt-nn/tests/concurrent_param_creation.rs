use std::sync::Arc;
use std::thread;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};

#[test]
fn concurrent_param_creation_same_name_one_wins() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<CpuBackend, f32>::new(backend, 42);

    let store_clone = store.clone();
    let handle1 = thread::spawn(move || {
        store_clone.param("shared_param", &[10, 10], Init::Zeros)
    });

    let store_clone = store.clone();
    let handle2 = thread::spawn(move || {
        store_clone.param("shared_param", &[10, 10], Init::Zeros)
    });

    let result1 = handle1.join().unwrap();
    let result2 = handle2.join().unwrap();

    let success_count = [result1.is_ok(), result2.is_ok()]
        .iter()
        .filter(|&&x| x)
        .count();

    assert_eq!(success_count, 1, "exactly one thread should succeed");

    let named = store.named_trainable();
    let shared_params: Vec<_> = named
        .iter()
        .filter(|(name, _)| name == "shared_param")
        .collect();
    
    assert_eq!(shared_params.len(), 1, "should have exactly one entry for 'shared_param'");
}

#[test]
fn concurrent_param_creation_different_names_all_succeed() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<CpuBackend, f32>::new(backend, 42);

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let store_clone = store.clone();
            thread::spawn(move || {
                store_clone.param(&format!("param_{}", i), &[5, 5], Init::Zeros)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    assert!(results.iter().all(|r| r.is_ok()), "all threads should succeed with different names");

    let named = store.named_trainable();
    assert_eq!(named.len(), 10, "should have exactly 10 params");

    let mut names: Vec<_> = named.iter().map(|(name, _)| name.as_str()).collect();
    names.sort();
    let expected: Vec<_> = (0..10).map(|i| format!("param_{}", i)).collect();
    let expected_refs: Vec<_> = expected.iter().map(|s| s.as_str()).collect();
    assert_eq!(names, expected_refs, "should have all expected param names");
}

#[test]
fn concurrent_buffer_creation_same_name_one_wins() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<CpuBackend, f32>::new(backend, 42);

    let store_clone = store.clone();
    let handle1 = thread::spawn(move || {
        store_clone.buffer("shared_buffer", &[10, 10], Init::Zeros)
    });

    let store_clone = store.clone();
    let handle2 = thread::spawn(move || {
        store_clone.buffer("shared_buffer", &[10, 10], Init::Zeros)
    });

    let result1 = handle1.join().unwrap();
    let result2 = handle2.join().unwrap();

    let success_count = [result1.is_ok(), result2.is_ok()]
        .iter()
        .filter(|&&x| x)
        .count();

    assert_eq!(success_count, 1, "exactly one thread should succeed");

    let named = store.named_buffers();
    let shared_buffers: Vec<_> = named
        .iter()
        .filter(|(name, _)| name == "shared_buffer")
        .collect();
    
    assert_eq!(shared_buffers.len(), 1, "should have exactly one entry for 'shared_buffer'");
}

#[test]
fn name_to_id_consistency_under_contention() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<CpuBackend, f32>::new(backend, 42);

    let handles: Vec<_> = (0..100)
        .map(|i| {
            let store_clone = store.clone();
            thread::spawn(move || {
                let _ = store_clone.param(&format!("param_{}", i % 20), &[3, 3], Init::Zeros);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let name_to_id = store.name_to_id();
    let id_to_name = store.id_to_name();

    for (name, id) in &name_to_id {
        assert_eq!(
            id_to_name.get(id).map(|s| s.as_str()),
            Some(name.as_str()),
            "name_to_id and id_to_name must be consistent"
        );
    }

    let named = store.named_trainable();
    assert_eq!(
        named.len(),
        name_to_id.len(),
        "named_trainable should have same count as name_to_id"
    );

    let mut names_from_named: Vec<_> = named.iter().map(|(n, _)| n.as_str()).collect();
    names_from_named.sort();
    names_from_named.dedup();
    
    assert_eq!(
        names_from_named.len(),
        named.len(),
        "named_trainable should not have duplicate names"
    );
}
