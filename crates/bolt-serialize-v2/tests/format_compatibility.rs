use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize_v2::{CheckpointMeta, CheckpointOptions, FormatKind, LoadOpts, save};

type B = CpuBackend;
type D = f32;

/// Test: SafeTensors format saves and loads correctly.
/// Expected: Data integrity is preserved with SafeTensors format.
#[test]
fn safetensors_format_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("safetensors");

    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.5, 2.5],
        &[2],
    )?)?;
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 1024 * 1024,
        },
    )?;

    // Verify format was saved
    use bolt_serialize_v2::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    assert_eq!(reader.info().format_kind, FormatKind::SafeTensors);

    // Load and verify
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    store_dst.seal();

    bolt_serialize_v2::load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    assert_eq!(w.tensor().to_vec()?, w2.tensor().to_vec()?);

    Ok(())
}

/// Test: Binary format saves and loads correctly.
/// Expected: Data integrity is preserved with Binary format.
#[test]
fn binary_format_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("binary");

    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[3.5, 4.5],
        &[2],
    )?)?;
    store_src.seal();

    save(
        &store_src,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::Binary,
            shard_max_bytes: 1024 * 1024,
        },
    )?;

    // Verify format was saved
    use bolt_serialize_v2::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    assert_eq!(reader.info().format_kind, FormatKind::Binary);

    // Load and verify
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store_dst.param("weight", &[2], Init::Zeros)?;
    store_dst.seal();

    bolt_serialize_v2::load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    assert_eq!(w.tensor().to_vec()?, w2.tensor().to_vec()?);

    Ok(())
}

/// Test: Format is automatically detected from manifest on load.
/// Expected: Loading a checkpoint automatically uses the correct format reader.
#[test]
fn format_auto_detection_on_load() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("format_detect");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let w = store.param("weight", &[2], Init::Zeros)?;
    w.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[5.0, 6.0],
        &[2],
    )?)?;
    store.seal();

    // Save with Binary format
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::Binary,
            shard_max_bytes: 1024 * 1024,
        },
    )?;

    // Load without specifying format - should auto-detect
    use bolt_serialize_v2::CheckpointReader;
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;

    // Should detect Binary format
    assert_eq!(reader.info().format_kind, FormatKind::Binary);

    // Should be able to read data
    let mut store2 = Store::<B, D>::new(backend.clone(), 2);
    let w2 = store2.param("weight", &[2], Init::Zeros)?;
    store2.seal();

    bolt_serialize_v2::load(&mut store2, &ckpt_dir, &LoadOpts::default())?;
    assert_eq!(w.tensor().to_vec()?, w2.tensor().to_vec()?);

    Ok(())
}
