use bolt_rng::ModelRng;
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, LoadOpts, load, save};

/// Test: RNG state roundtrip preserves randomness state.
/// Expected: After loading, RNG produces the same sequence as before saving.
#[test]
fn rng_state_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("rng_roundtrip");

    let mut rng_src = ModelRng::from_seed(42);
    let state_before = rng_src.state();

    // Use RNG to advance state
    let _ = rng_src.forward_rngs();
    let _ = rng_src.data_rng_for_epoch(5);

    save(
        &rng_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    let mut rng_dst = ModelRng::from_seed(999); // Different seed initially
    load(&mut rng_dst, &ckpt_dir, &LoadOpts::default())?;

    // State should match
    assert_eq!(rng_src.state(), rng_dst.state());

    // Both should produce same next values
    let next_src = rng_src.forward_rngs();
    let next_dst = rng_dst.forward_rngs();
    assert_eq!(next_src.dropout.state(), next_dst.dropout.state());
    assert_eq!(next_src.data.state(), next_dst.data.state());
    assert_eq!(next_src.noise.state(), next_dst.noise.state());

    Ok(())
}

/// Test: RNG can be saved and loaded with different initial seeds.
/// Expected: Loaded RNG state overrides initial seed, producing correct sequence.
#[test]
fn rng_load_overrides_initial_seed() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("rng_seed_override");

    let mut rng_src = ModelRng::from_seed(100);
    let _ = rng_src.forward_rngs();
    let state_saved = rng_src.state();

    save(
        &rng_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Create new RNG with different seed
    let mut rng_dst = ModelRng::from_seed(200);

    // Before loading, states should be different
    assert_ne!(rng_src.state(), rng_dst.state());

    load(&mut rng_dst, &ckpt_dir, &LoadOpts::default())?;

    // After loading, states should match
    assert_eq!(state_saved, rng_dst.state());

    Ok(())
}
