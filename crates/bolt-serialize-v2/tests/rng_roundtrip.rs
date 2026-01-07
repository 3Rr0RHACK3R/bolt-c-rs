use bolt_rng::RngKey;
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

/// Test: RNG key roundtrip preserves key value.
/// Expected: After loading, RNG key produces the same sequence as before saving.
#[test]
fn rng_key_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("rng_roundtrip");

    let key_src = RngKey::from_seed(42);
    
    // Generate some values before saving
    let mut seq_before = key_src.into_seq();
    let val1 = seq_before.next_u64();
    let val2 = seq_before.next_u64();

    save(
        &key_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    let mut key_dst = RngKey::from_seed(999); // Different seed initially
    assert_ne!(key_src.key(), key_dst.key());
    
    load(&mut key_dst, &ckpt_dir, &LoadOpts::default())?;

    // Keys should match after loading
    assert_eq!(key_src.key(), key_dst.key());

    // Both should produce same sequence
    let mut seq_src = key_src.into_seq();
    let mut seq_dst = key_dst.into_seq();
    
    // Should match the values generated before saving
    assert_eq!(seq_src.next_u64(), val1);
    assert_eq!(seq_dst.next_u64(), val1);
    assert_eq!(seq_src.next_u64(), val2);
    assert_eq!(seq_dst.next_u64(), val2);

    Ok(())
}

/// Test: RNG key can be saved and loaded with different initial seeds.
/// Expected: Loaded key overrides initial seed, producing correct sequence.
#[test]
fn rng_key_load_overrides_initial_seed() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("rng_seed_override");

    let key_src = RngKey::from_seed(100);
    let key_before = key_src.key();

    save(
        &key_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Create new key with different seed
    let mut key_dst = RngKey::from_seed(200);

    // Before loading, keys should be different
    assert_ne!(key_before, key_dst.key());

    load(&mut key_dst, &ckpt_dir, &LoadOpts::default())?;

    // After loading, keys should match
    assert_eq!(key_before, key_dst.key());

    // Both should produce same sequence
    let mut seq_src = key_src.into_seq();
    let mut seq_dst = key_dst.into_seq();
    for _ in 0..100 {
        assert_eq!(seq_src.next_u64(), seq_dst.next_u64());
    }

    Ok(())
}
