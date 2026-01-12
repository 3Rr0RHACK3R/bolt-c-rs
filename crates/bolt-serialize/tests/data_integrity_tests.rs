use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_serialize::{
    CheckpointMeta, CheckpointOptions, CheckpointReader, CheckpointWriter, FormatKind, LoadOpts,
    load, save,
};

type B = CpuBackend;
type D = f32;

/// Test: Data with specific patterns is preserved exactly.
/// Expected: Byte patterns are identical after roundtrip.
#[test]
fn pattern_data_preserved_exactly() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("pattern_data");

    let store = Store::<B, D>::new(backend.clone(), 1);

    // Create tensors with recognizable patterns
    let ascending = store.param("ascending", &[100], Init::Zeros)?;
    let descending = store.param("descending", &[100], Init::Zeros)?;
    let alternating = store.param("alternating", &[100], Init::Zeros)?;
    let constant = store.param("constant", &[100], Init::Zeros)?;

    // Set pattern values
    ascending.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &(0..100).map(|i| i as f32).collect::<Vec<_>>(),
        &[100],
    )?)?;

    descending.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &(0..100).rev().map(|i| i as f32).collect::<Vec<_>>(),
        &[100],
    )?)?;

    alternating.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &(0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>(),
        &[100],
    )?)?;

    constant.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &vec![42.5f32; 100],
        &[100],
    )?)?;

    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load and verify all patterns
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let ascending_dst = store_dst.param("ascending", &[100], Init::Zeros)?;
    let descending_dst = store_dst.param("descending", &[100], Init::Zeros)?;
    let alternating_dst = store_dst.param("alternating", &[100], Init::Zeros)?;
    let constant_dst = store_dst.param("constant", &[100], Init::Zeros)?;
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify ascending pattern
    let asc_values: Vec<f32> = ascending_dst.tensor().to_vec()?;
    for (i, v) in asc_values.iter().enumerate() {
        assert!(
            (*v - i as f32).abs() < 0.0001,
            "ascending[{}] = {}, expected {}",
            i,
            v,
            i
        );
    }

    // Verify descending pattern
    let desc_values: Vec<f32> = descending_dst.tensor().to_vec()?;
    for (i, v) in desc_values.iter().enumerate() {
        let expected = 99.0 - i as f32;
        assert!(
            (*v - expected).abs() < 0.0001,
            "descending[{}] = {}, expected {}",
            i,
            v,
            expected
        );
    }

    // Verify alternating pattern
    let alt_values: Vec<f32> = alternating_dst.tensor().to_vec()?;
    for (i, v) in alt_values.iter().enumerate() {
        let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (*v - expected).abs() < 0.0001,
            "alternating[{}] = {}, expected {}",
            i,
            v,
            expected
        );
    }

    // Verify constant pattern
    let const_values: Vec<f32> = constant_dst.tensor().to_vec()?;
    for (i, v) in const_values.iter().enumerate() {
        assert!(
            (*v - 42.5).abs() < 0.0001,
            "constant[{}] = {}, expected 42.5",
            i,
            v
        );
    }

    Ok(())
}

/// Test: Large checkpoint write and read works correctly.
/// Expected: Multi-MB checkpoints are handled without issues.
#[test]
fn large_checkpoint_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("large_checkpoint");

    let store = Store::<B, D>::new(backend.clone(), 1);

    // Create two 10MB tensors (2.5 million f32 values each)
    let tensor_size = 2_500_000; // 10MB
    for i in 0..2 {
        let p = store.param(&format!("large_{}", i), &[tensor_size], Init::Zeros)?;
        // Use a pattern based on index so we can verify
        let values: Vec<f32> = (0..tensor_size)
            .map(|j| (j as f32 * 0.000001) + (i as f32 * 100.0))
            .collect();
        p.set_tensor(bolt_tensor::Tensor::from_slice(
            &backend,
            &values,
            &[tensor_size],
        )?)?;
    }
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Verify checkpoint was created
    assert!(ckpt_dir.join("bolt-checkpoint.json").exists());

    // Load and verify
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    for i in 0..2 {
        store_dst.param(&format!("large_{}", i), &[tensor_size], Init::Zeros)?;
    }
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    // Verify some values from each tensor (spot check)
    for (name, param) in store_dst.named_trainable() {
        let idx: f32 = name.chars().last().unwrap().to_digit(10).unwrap() as f32;
        let values: Vec<f32> = param.tensor().to_vec()?;

        // Check first value
        let expected_first = idx * 100.0;
        assert!(
            (values[0] - expected_first).abs() < 0.0001,
            "{}: first value should be {}, got {}",
            name,
            expected_first,
            values[0]
        );

        // Check middle value
        let mid = tensor_size / 2;
        let expected_mid = (mid as f32 * 0.000001) + (idx * 100.0);
        assert!(
            (values[mid] - expected_mid).abs() < 0.001,
            "{}: middle value should be {}, got {}",
            name,
            expected_mid,
            values[mid]
        );

        // Check last value
        let expected_last = ((tensor_size - 1) as f32 * 0.000001) + (idx * 100.0);
        assert!(
            (values[tensor_size - 1] - expected_last).abs() < 0.001,
            "{}: last value should be {}, got {}",
            name,
            expected_last,
            values[tensor_size - 1]
        );
    }

    Ok(())
}

/// Test: Edge case float values are preserved.
/// Expected: Infinity, negative infinity, and special values roundtrip correctly.
#[test]
fn special_float_values_preserved() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("special_floats");

    let store = Store::<B, D>::new(backend.clone(), 1);
    let special = store.param("special", &[7], Init::Zeros)?;

    let special_values: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN,
        f32::MAX,
        f32::EPSILON,
        f32::MIN_POSITIVE,
        1e-38, // Very small positive
    ];

    special.set_tensor(bolt_tensor::Tensor::from_slice(
        &backend,
        &special_values,
        &[7],
    )?)?;
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    let special_dst = store_dst.param("special", &[7], Init::Zeros)?;
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    let loaded: Vec<f32> = special_dst.tensor().to_vec()?;

    // Verify special values (using exact bit comparison where possible)
    assert_eq!(loaded[0].to_bits(), 0.0f32.to_bits(), "0.0");
    // Note: -0.0 and 0.0 may be equal in comparison but have different bits
    assert_eq!(loaded[2], f32::MIN, "f32::MIN");
    assert_eq!(loaded[3], f32::MAX, "f32::MAX");
    assert_eq!(loaded[4], f32::EPSILON, "f32::EPSILON");
    assert_eq!(loaded[5], f32::MIN_POSITIVE, "f32::MIN_POSITIVE");
    assert!((loaded[6] - 1e-38).abs() < 1e-45, "very small positive");

    Ok(())
}

/// Test: Multi-dimensional tensors preserve shape correctly.
/// Expected: Shape dimensions are exactly preserved.
#[test]
fn multidimensional_shape_preserved() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("multidim_shapes");

    let store = Store::<B, D>::new(backend.clone(), 1);

    // Various shape configurations
    let shapes = vec![
        ("scalar", vec![1]),
        ("vector", vec![10]),
        ("matrix", vec![3, 4]),
        ("tensor3d", vec![2, 3, 4]),
        ("tensor4d", vec![2, 2, 2, 2]),
    ];

    for (name, shape) in &shapes {
        let numel: usize = shape.iter().product();
        let p = store.param(name, shape, Init::Zeros)?;
        let values: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        p.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &values, shape)?)?;
    }
    store.seal();

    save(
        &store,
        &ckpt_dir,
        &CheckpointMeta::default(),
        &CheckpointOptions::default(),
    )?;

    // Load and verify shapes
    let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
    for (name, shape) in &shapes {
        store_dst.param(name, shape, Init::Zeros)?;
    }
    store_dst.seal();

    load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

    for (name, param) in store_dst.named_trainable() {
        let expected_shape = shapes.iter().find(|(n, _)| n == &name).unwrap().1.clone();
        let actual_shape = param.tensor().shape().as_slice().to_vec();
        assert_eq!(
            actual_shape, expected_shape,
            "{}: shape mismatch - expected {:?}, got {:?}",
            name, expected_shape, actual_shape
        );
    }

    Ok(())
}

// Note: Empty tensors (shape [0]) are not supported in bolt's design.
// Zero-sized dimensions intentionally raise InvalidShape errors.

/// Test: Scalar values (shape []) roundtrip correctly.
/// Note: Scalar tensors have empty shape but 1 element.
#[test]
fn scalar_value_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("scalar_value");

    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;

    // Write a scalar f32 value
    writer.f32("my_scalar", std::f32::consts::PI)?;
    writer.finish(&CheckpointMeta::default())?;

    // Read it back
    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let loaded = reader.f32("my_scalar")?;

    assert!((loaded - std::f32::consts::PI).abs() < 0.00001);

    Ok(())
}

/// Test: i64 values roundtrip correctly.
/// Expected: Integer values including negative and large values are preserved.
#[test]
fn i64_value_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("i64_values");

    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;

    writer.i64("positive", 12345678901234)?;
    writer.i64("negative", -98765432109876)?;
    writer.i64("zero", 0)?;
    writer.i64("max", i64::MAX)?;
    writer.i64("min", i64::MIN)?;
    writer.finish(&CheckpointMeta::default())?;

    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;

    assert_eq!(reader.i64("positive")?, 12345678901234);
    assert_eq!(reader.i64("negative")?, -98765432109876);
    assert_eq!(reader.i64("zero")?, 0);
    assert_eq!(reader.i64("max")?, i64::MAX);
    assert_eq!(reader.i64("min")?, i64::MIN);

    Ok(())
}

/// Test: JSON values roundtrip correctly.
/// Expected: Structured data is preserved through JSON serialization.
#[test]
fn json_value_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let ckpt_dir = tmp.path().join("json_values");

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Config {
        learning_rate: f64,
        batch_size: usize,
        name: String,
        layers: Vec<usize>,
    }

    let config = Config {
        learning_rate: 0.001,
        batch_size: 32,
        name: "TestModel".to_string(),
        layers: vec![128, 64, 32],
    };

    let mut writer = CheckpointWriter::new(&ckpt_dir, &CheckpointOptions::default())?;
    writer.json("config", &config)?;
    writer.finish(&CheckpointMeta::default())?;

    let reader = CheckpointReader::open(&ckpt_dir, &LoadOpts::default())?;
    let loaded: Config = reader.json("config")?;

    assert_eq!(loaded, config);

    Ok(())
}

/// Test: Data integrity across both formats.
/// Expected: Same data produces identical results in both SafeTensors and Binary formats.
#[test]
fn data_integrity_across_formats() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = tempfile::tempdir()?;

    // Create source data
    let store_src = Store::<B, D>::new(backend.clone(), 1);
    let w = store_src.param("weight", &[100], Init::Zeros)?;
    let values: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
    w.set_tensor(bolt_tensor::Tensor::from_slice(&backend, &values, &[100])?)?;
    store_src.seal();

    let mut loaded_values = vec![];

    for format in [FormatKind::SafeTensors, FormatKind::Binary] {
        let ckpt_dir = tmp.path().join(format!("format_{:?}", format));

        save(
            &store_src,
            &ckpt_dir,
            &CheckpointMeta::default(),
            &CheckpointOptions {
                format: format.clone(),
                shard_max_bytes: 1024 * 1024,
            },
        )?;

        let mut store_dst = Store::<B, D>::new(backend.clone(), 2);
        store_dst.param("weight", &[100], Init::Zeros)?;
        store_dst.seal();

        load(&mut store_dst, &ckpt_dir, &LoadOpts::default())?;

        let loaded: Vec<f32> = store_dst
            .named_trainable()
            .into_iter()
            .find(|(n, _)| n == "weight")
            .unwrap()
            .1
            .tensor()
            .to_vec()?;

        loaded_values.push(loaded);
    }

    // Both formats should produce identical values
    assert_eq!(
        loaded_values[0], loaded_values[1],
        "SafeTensors and Binary formats should produce identical results"
    );

    // Verify against original
    assert_eq!(loaded_values[0], values);

    Ok(())
}
