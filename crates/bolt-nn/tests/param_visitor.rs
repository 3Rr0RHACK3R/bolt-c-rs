use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::Store;
use bolt_nn::layers::Linear;

type B = CpuBackend;
type D = f32;

#[test]
fn store_named_trainable_is_deterministic_and_keyed() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend, 0);

    let _ = Linear::init(&store.sub("a"), 2, 3, true).unwrap();
    let _ = Linear::init(&store.sub("b"), 2, 3, false).unwrap();

    let keys: Vec<String> = store
        .named_trainable()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    // Order is determined by ParamId (insertion order): weight created before bias
    assert_eq!(
        keys,
        vec![
            "a.weight".to_string(),
            "a.bias".to_string(),
            "b.weight".to_string(),
        ]
    );
}
