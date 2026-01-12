use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize::{
    CheckpointMeta, CheckpointOptions, CheckpointReader, FormatKind, LoadOpts, load, save,
};

type B = CpuBackend;
type D = f32;

/// Test: Many records create multiple shards when shard_max_bytes is set.
/// Expected: Checkpoint is split into multiple shard files.
#[test]
fn many_records_creates_shards() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("many_shards");

    // Create 10 tensors, each ~1KB (250 f32 values)
    let store = Store::<B, D>::new(backend.clone(), 1);
    for i in 0..10 {
        let p = store.param(&format!("tensor_{:03}", i), &[250], Init::Zeros)?;
        // Each tensor is 250 * 4 = 1000 bytes
        let values: Vec<f32> = (0..250).map(|v| v as f32 + i as f32 * 1000.0).collect();
        p.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &values, &[250])?)?;
    }
    store.seal();

    // Set shard_max_bytes to 2KB to force multiple shards (each tensor is 1KB)
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 2000, // 2KB max per shard
        },
    )?;

    // Verify checkpoint was created
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    // Should have multiple shards (10 tensors * 1KB each / 2KB per shard = ~5 shards)
    assert!(
        info.shard_count > 1,
        "Expected multiple shards, got {}",
        info.shard_count
    );

    // Verify all data can be read back correctly
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    for i in 0..10 {
        store_dst.param(&format!("tensor_{:03}", i), &[250], Init::Zeros)?;
    }
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify each tensor has correct values
    for (name, param) in store_dst.named_trainable() {
        let values: Vec<f32> = param.tensor().to_vec()?;
        assert_eq!(values.len(), 250, "Tensor {} has wrong size", name);

        // Extract index from name (tensor_XXX)
        let idx: usize = name[7..10].parse()?;
        let expected_first = idx as f32 * 1000.0;
        assert!(
            (values[0] - expected_first).abs() < 0.001,
            "First value of {} should be {}, got {}",
            name,
            expected_first,
            values[0]
        );
    }

    Ok(())
}

/// Test: Single small checkpoint uses single shard.
/// Expected: Only one shard file is created.
#[test]
fn small_checkpoint_single_shard() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("single_shard");

    // Create a few small tensors
    let store = Store::<B, D>::new(backend.clone(), 1);
    let _w = store.param("weight", &[10], Init::Zeros)?;
    let _b = store.param("bias", &[5], Init::Zeros)?;
    store.seal();

    // Use large shard size so everything fits in one shard
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 1024 * 1024, // 1MB
        },
    )?;

    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    // Should have exactly one shard
    assert_eq!(
        info.shard_count, 1,
        "Expected single shard, got {}",
        info.shard_count
    );

    Ok(())
}

/// Test: Sharding works correctly with both SafeTensors and Binary formats.
/// Expected: Both formats handle sharding consistently.
#[test]
fn sharding_works_with_all_formats() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;

    for format in [FormatKind::SafeTensors, FormatKind::Binary] {
        let ckpt_dir = tmp.path().join(format!("sharding_{:?}", format));

        // Create 5 tensors, each 500 bytes (125 f32 values)
        let store = Store::<B, D>::new(backend.clone(), 1);
        for i in 0..5 {
            let p = store.param(&format!("tensor_{}", i), &[125], Init::Zeros)?;
            let values: Vec<f32> = (0..125).map(|v| v as f32 * (i + 1) as f32).collect();
            p.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &values, &[125])?)?;
        }
        store.seal();

        // Force multiple shards
        save(
            &store,
            &ckpt_dir,
            &CheckpointMeta::default(),
            &CheckpointOptions {
                format: format.clone(),
                shard_max_bytes: 600, // ~1.2 tensors per shard
            },
        )?;

        // Verify format and shard count
        let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
        let info = reader.info();

        assert_eq!(info.format_kind, format);
        assert!(
            info.shard_count > 1,
            "Format {:?} should have multiple shards, got {}",
            format,
            info.shard_count
        );

        // Verify data integrity
        let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
        for i in 0..5 {
            store_dst.param(&format!("tensor_{}", i), &[125], Init::Zeros)?;
        }
        store_dst.seal();

        load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

        // Verify each tensor
        for (name, param) in store_dst.named_trainable() {
            let idx: usize = name.chars().last().unwrap().to_digit(10).unwrap() as usize;
            let values: Vec<f32> = param.tensor().to_vec()?;
            let expected_first = 0.0;
            let expected_second = 1.0 * (idx + 1) as f32;
            assert!(
                (values[0] - expected_first).abs() < 0.001,
                "{}: expected first value {}, got {}",
                name,
                expected_first,
                values[0]
            );
            assert!(
                (values[1] - expected_second).abs() < 0.001,
                "{}: expected second value {}, got {}",
                name,
                expected_second,
                values[1]
            );
        }
    }

    Ok(())
}

/// Test: Very large tensor that exceeds shard_max_bytes is handled correctly.
/// Expected: A single large tensor gets its own shard even if it exceeds the limit.
#[test]
fn oversized_tensor_gets_own_shard() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("oversized");

    // Create one large tensor (4KB) and one small tensor (40 bytes)
    let store = Store::<B, D>::new(backend.clone(), 1);
    let large = store.param("large", &[1000], Init::Zeros)?; // 4000 bytes
    let small = store.param("small", &[10], Init::Zeros)?; // 40 bytes

    let large_values: Vec<f32> = (0..1000).map(|v| v as f32 * 0.001).collect();
    large.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &large_values,
        &[1000],
    )?)?;
    small.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &[1.0; 10],
        &[10],
    )?)?;
    store.seal();

    // Set shard_max_bytes smaller than the large tensor
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 100, // Much smaller than the large tensor
        },
    )?;

    // Load and verify both tensors are intact
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let large_dst = store_dst.param("large", &[1000], Init::Zeros)?;
    let small_dst = store_dst.param("small", &[10], Init::Zeros)?;
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify large tensor values
    let large_loaded: Vec<f32> = large_dst.tensor().to_vec()?;
    assert_eq!(large_loaded.len(), 1000);
    assert!((large_loaded[0] - 0.0).abs() < 0.0001);
    assert!((large_loaded[500] - 0.5).abs() < 0.0001);
    assert!((large_loaded[999] - 0.999).abs() < 0.0001);

    // Verify small tensor values
    let small_loaded: Vec<f32> = small_dst.tensor().to_vec()?;
    assert_eq!(small_loaded, vec![1.0; 10]);

    Ok(())
}

/// Test: Checkpoint info correctly reports shard count.
/// Expected: The shard_count in checkpoint info matches actual number of shards.
#[test]
fn checkpoint_info_reports_shard_count() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("shard_count_info");

    // Create tensors that will be split across exactly 3 shards
    let store = Store::<B, D>::new(backend.clone(), 1);
    for i in 0..6 {
        let p = store.param(&format!("tensor_{}", i), &[100], Init::Zeros)?; // 400 bytes each
        let values: Vec<f32> = vec![i as f32; 100];
        p.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &values, &[100])?)?;
    }
    store.seal();

    // Set shard size to hold 2 tensors each (800 bytes)
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 850, // ~2 tensors per shard
        },
    )?;

    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let info = reader.info();

    // Should have 3 shards (6 tensors / 2 per shard)
    assert!(
        info.shard_count >= 3,
        "Expected at least 3 shards, got {}",
        info.shard_count
    );

    Ok(())
}

/// Test: Records from sharded checkpoint maintain correct order/access.
/// Expected: All records are accessible regardless of which shard they're in.
#[test]
fn sharded_records_accessible_by_name() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("sharded_access");

    // Create records with various names that will be distributed across shards
    let store = Store::<B, D>::new(backend.clone(), 1);

    // Use names that would sort differently to test index correctness
    let names = ["z_last", "a_first", "m_middle", "b_second", "y_penultimate"];
    for name in &names {
        let p = store.param(name, &[50], Init::Zeros)?; // 200 bytes each
        let values: Vec<f32> = name.as_bytes().iter().map(|&b| b as f32).collect();
        let mut padded = vec![0.0f32; 50];
        for (i, v) in values.into_iter().enumerate() {
            if i < 50 {
                padded[i] = v;
            }
        }
        p.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &padded, &[50])?)?;
    }
    store.seal();

    // Force multiple shards
    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 300, // ~1-2 records per shard
        },
    )?;

    // Load and verify each record by name
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    for name in &names {
        store_dst.param(name, &[50], Init::Zeros)?;
    }
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify each record has correct first bytes
    for (name, param) in store_dst.named_trainable() {
        let values: Vec<f32> = param.tensor().to_vec()?;
        let expected_first = name.as_bytes()[0] as f32;
        assert!(
            (values[0] - expected_first).abs() < 0.001,
            "Record {} first value should be {}, got {}",
            name,
            expected_first,
            values[0]
        );
    }

    Ok(())
}
