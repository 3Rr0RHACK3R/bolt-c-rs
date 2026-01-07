use bolt_rng::RngKey;
use std::collections::HashSet;

#[test]
fn rng_key_same_seed_same_sequence() {
    let mut seq1 = RngKey::from_seed(42).into_seq();
    let mut seq2 = RngKey::from_seed(42).into_seq();
    for _ in 0..10_000 {
        assert_eq!(seq1.next_u64(), seq2.next_u64());
    }
}

#[test]
fn rng_key_split_produces_independent_keys() {
    let parent = RngKey::from_seed(123);
    let (k1, k2) = parent.split();
    
    let mut seq1 = k1.into_seq();
    let mut seq2 = k2.into_seq();
    assert_ne!(seq1.next_u64(), seq2.next_u64());
}

#[test]
fn rng_key_fold_in_is_stable() {
    let base = RngKey::from_seed(7);
    let mut seq1 = base.fold_in(0).into_seq();
    let mut seq2 = base.fold_in(0).into_seq();
    for _ in 0..1024 {
        assert_eq!(seq1.next_u64(), seq2.next_u64());
    }
}

#[test]
fn rng_key_derive_is_stable() {
    let base = RngKey::from_seed(999);
    let mut seq1 = base.derive("test").into_seq();
    let mut seq2 = base.derive("test").into_seq();
    for _ in 0..1024 {
        assert_eq!(seq1.next_u64(), seq2.next_u64());
    }
}

#[test]
fn rng_key_derive_path_is_stable() {
    let base = RngKey::from_seed(555);
    let mut seq1 = base.derive_path(&["a", "b", "c"]).into_seq();
    let mut seq2 = base.derive_path(&["a", "b", "c"]).into_seq();
    for _ in 0..1024 {
        assert_eq!(seq1.next_u64(), seq2.next_u64());
    }
}

#[test]
fn rng_key_split_n_produces_independent_keys() {
    let parent = RngKey::from_seed(777);
    let keys = parent.split_n(10);
    
    assert_eq!(keys.len(), 10);
    let mut seqs: Vec<_> = keys.iter().map(|k| k.into_seq()).collect();
    
    // All sequences should be different
    let first_values: Vec<u64> = seqs.iter_mut().map(|s| s.next_u64()).collect();
    for i in 0..first_values.len() {
        for j in (i + 1)..first_values.len() {
            assert_ne!(first_values[i], first_values[j]);
        }
    }
}

#[test]
fn rng_key_split_n_is_stable() {
    let parent = RngKey::from_seed(888);
    let keys = parent.split_n(5);
    
    assert_eq!(keys.iter().map(|k| k.key()).collect::<HashSet<_>>().len(), keys.len());
    
    // Verify stability: same seed should produce same keys
    let parent2 = RngKey::from_seed(888);
    let keys2 = parent2.split_n(5);
    for (k1, k2) in keys.iter().zip(keys2.iter()) {
        assert_eq!(k1.key(), k2.key());
    }
}

#[test]
fn rng_key_different_derive_tags_produce_different_streams() {
    let base = RngKey::from_seed(111);
    let mut seq_a = base.derive("a").into_seq();
    let mut seq_b = base.derive("b").into_seq();
    
    // Different tags should produce different sequences
    assert_ne!(seq_a.next_u64(), seq_b.next_u64());
}

#[test]
fn rng_key_order_independence() {
    // Same seed + same path should produce same result regardless of creation order
    let root1 = RngKey::from_seed(222);
    let root2 = RngKey::from_seed(222);
    
    // Create keys in different orders
    let key1a = root1.derive("init").derive_path(&["params", "layer1", "w"]);
    let key1b = root1.derive("init").derive_path(&["params", "layer1", "b"]);
    
    let key2b = root2.derive("init").derive_path(&["params", "layer1", "b"]);
    let key2a = root2.derive("init").derive_path(&["params", "layer1", "w"]);
    
    let mut seq1a = key1a.into_seq();
    let mut seq1b = key1b.into_seq();
    let mut seq2a = key2a.into_seq();
    let mut seq2b = key2b.into_seq();
    
    // Same path should produce same sequence regardless of order
    for _ in 0..100 {
        assert_eq!(seq1a.next_u64(), seq2a.next_u64());
        assert_eq!(seq1b.next_u64(), seq2b.next_u64());
    }
}

#[test]
fn rng_key_fold_in_different_values_produces_different_streams() {
    let base = RngKey::from_seed(333);
    let mut seq0 = base.fold_in(0).into_seq();
    let mut seq1 = base.fold_in(1).into_seq();
    let mut seq2 = base.fold_in(2).into_seq();
    
    let v0 = seq0.next_u64();
    let v1 = seq1.next_u64();
    let v2 = seq2.next_u64();
    
    assert_ne!(v0, v1);
    assert_ne!(v1, v2);
    assert_ne!(v0, v2);
}

#[test]
fn rng_seq_distribution_uniform() {
    let mut seq = RngKey::from_seed(444).into_seq();
    let n = 10000;
    let mut sum = 0.0;
    
    for _ in 0..n {
        sum += seq.next_f64_01();
    }
    
    let mean = sum / n as f64;
    // Mean should be close to 0.5 for uniform [0, 1)
    assert!((mean - 0.5).abs() < 0.02, "Mean should be close to 0.5, got {}", mean);
}

#[test]
fn rng_seq_gen_range_uniform() {
    let mut seq = RngKey::from_seed(555).into_seq();
    let range = 10..20;
    let n = 10000;
    let mut counts = vec![0; 10];
    
    for _ in 0..n {
        let val = seq.gen_range(range.clone());
        assert!(val >= range.start && val < range.end);
        counts[val - range.start] += 1;
    }
    
    // Each value should appear roughly equally often
    let expected: i32 = n / 10;
    for count in counts {
        assert!((count - expected).abs() < expected / 5);
    }
}

#[test]
fn rng_key_concurrent_usage() {
    // Simulate concurrent usage by splitting keys
    let root = RngKey::from_seed(666);
    let keys = root.split_n(4);
    
    let results: Vec<Vec<u64>> = keys.iter().map(|k| {
        let mut seq = k.into_seq();
        (0..100).map(|_| seq.next_u64()).collect()
    }).collect();
    
    // All sequences should be different
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert_ne!(results[i], results[j]);
        }
    }
}

#[test]
fn rng_key_deterministic_across_recreations() {
    let seed = 777;
    
    // Create two keys from same seed
    let root1 = RngKey::from_seed(seed);
    let root2 = RngKey::from_seed(seed);
    
    // Derive same paths
    let init1 = root1.derive("init");
    let init2 = root2.derive("init");
    
    let step1 = root1.derive("step").fold_in(5);
    let step2 = root2.derive("step").fold_in(5);
    
    let mut seq1a = init1.into_seq();
    let mut seq2a = init2.into_seq();
    let mut seq1b = step1.into_seq();
    let mut seq2b = step2.into_seq();
    
    // Should produce identical sequences
    for _ in 0..100 {
        assert_eq!(seq1a.next_u64(), seq2a.next_u64());
        assert_eq!(seq1b.next_u64(), seq2b.next_u64());
    }
}
