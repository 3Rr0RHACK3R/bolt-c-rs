use bolt_rng::{ModelRng, RngStream, RngStreams};

#[test]
fn rng_stream_same_seed_same_sequence() {
    let mut a = RngStream::from_seed(42);
    let mut b = RngStream::from_seed(42);
    for _ in 0..10_000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_stream_split_produces_independent_child() {
    let mut parent = RngStream::from_seed(123);
    let mut child = parent.split();
    assert_ne!(parent.next_u64(), child.next_u64());
}

#[test]
fn rng_stream_fold_in_is_stable() {
    let base = RngStream::from_seed(7);
    let mut a = base.fold_in(0);
    let mut b = base.fold_in(0);
    for _ in 0..1024 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_streams_split2_is_stable() {
    let mut s1 = RngStreams::from_seed(999);
    let mut s2 = RngStreams::from_seed(999);
    let (a1, b1) = s1.split2();
    let (a2, b2) = s2.split2();

    let mut a1 = a1;
    let mut b1 = b1;
    let mut a2 = a2;
    let mut b2 = b2;

    for _ in 0..2048 {
        assert_eq!(a1.dropout.next_u64(), a2.dropout.next_u64());
        assert_eq!(b1.dropout.next_u64(), b2.dropout.next_u64());
    }
}

#[test]
fn model_rng_same_seed_same_init_rng() {
    let mut rng1 = ModelRng::from_seed(42);
    let mut rng2 = ModelRng::from_seed(42);

    let mut init1 = rng1.init_rng();
    let mut init2 = rng2.init_rng();

    // Same seed should produce same init RNG stream
    for _ in 0..1000 {
        assert_eq!(init1.next_u64(), init2.next_u64());
    }
}

#[test]
fn model_rng_same_seed_same_forward_rngs() {
    let mut rng1 = ModelRng::from_seed(123);
    let mut rng2 = ModelRng::from_seed(123);

    let mut forward1 = rng1.forward_rngs();
    let mut forward2 = rng2.forward_rngs();

    // Same seed should produce same forward RNG streams
    for _ in 0..1000 {
        assert_eq!(forward1.dropout.next_u64(), forward2.dropout.next_u64());
        assert_eq!(forward1.data.next_u64(), forward2.data.next_u64());
        assert_eq!(forward1.noise.next_u64(), forward2.noise.next_u64());
    }
}

#[test]
fn model_rng_same_seed_same_data_rng_for_epoch() {
    let rng1 = ModelRng::from_seed(456);
    let rng2 = ModelRng::from_seed(456);

    // Same seed and epoch should produce same data RNG
    for epoch in 0..10 {
        let mut data1 = rng1.data_rng_for_epoch(epoch);
        let mut data2 = rng2.data_rng_for_epoch(epoch);

        for _ in 0..1000 {
            assert_eq!(data1.next_u64(), data2.next_u64());
        }
    }
}

#[test]
fn model_rng_different_epochs_different_data_rngs() {
    let rng = ModelRng::from_seed(789);

    let mut epoch0 = rng.data_rng_for_epoch(0);
    let mut epoch1 = rng.data_rng_for_epoch(1);
    let mut epoch2 = rng.data_rng_for_epoch(2);

    // Different epochs should produce different RNG streams
    let val0 = epoch0.next_u64();
    let val1 = epoch1.next_u64();
    let val2 = epoch2.next_u64();

    assert_ne!(val0, val1);
    assert_ne!(val1, val2);
    assert_ne!(val0, val2);
}

#[test]
fn model_rng_forward_rngs_produces_independent_streams() {
    let mut rng = ModelRng::from_seed(999);

    let mut forward1 = rng.forward_rngs();
    let mut forward2 = rng.forward_rngs();
    let mut forward3 = rng.forward_rngs();

    // Each forward_rngs() call should produce independent streams
    let val1 = forward1.dropout.next_u64();
    let val2 = forward2.dropout.next_u64();
    let val3 = forward3.dropout.next_u64();

    assert_ne!(val1, val2);
    assert_ne!(val2, val3);
    assert_ne!(val1, val3);
}

#[test]
fn model_rng_streams_are_independent() {
    let mut rng = ModelRng::from_seed(111);

    // Get one value from each stream type
    let init_val = rng.init_rng().next_u64();
    let forward_val = rng.forward_rngs().dropout.next_u64();
    let data_val = rng.data_rng_for_epoch(0).next_u64();

    // All streams should be independent
    assert_ne!(init_val, forward_val);
    assert_ne!(forward_val, data_val);
    assert_ne!(init_val, data_val);
}

#[test]
fn model_rng_deterministic_across_recreations() {
    let seed = 555;

    // Create two ModelRngs from same seed
    let mut rng1 = ModelRng::from_seed(seed);
    let mut rng2 = ModelRng::from_seed(seed);

    // Get init RNGs - should be identical
    let mut init1 = rng1.init_rng();
    let mut init2 = rng2.init_rng();
    assert_eq!(init1.next_u64(), init2.next_u64());

    // Get forward RNGs - should be identical
    let mut forward1 = rng1.forward_rngs();
    let mut forward2 = rng2.forward_rngs();
    assert_eq!(forward1.dropout.next_u64(), forward2.dropout.next_u64());

    // Get data RNGs for same epoch - should be identical
    let mut data1 = rng1.data_rng_for_epoch(5);
    let mut data2 = rng2.data_rng_for_epoch(5);
    assert_eq!(data1.next_u64(), data2.next_u64());
}

#[test]
fn model_rng_data_rng_epoch_folding_is_stable() {
    let rng = ModelRng::from_seed(777);

    // Same epoch should always produce same RNG stream
    for epoch in 0..5 {
        let mut data1 = rng.data_rng_for_epoch(epoch);
        let mut data2 = rng.data_rng_for_epoch(epoch);

        for _ in 0..100 {
            assert_eq!(data1.next_u64(), data2.next_u64());
        }
    }
}
