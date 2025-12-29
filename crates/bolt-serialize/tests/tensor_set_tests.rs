use std::fs;
use std::sync::Arc;

use bolt_core::DType;
use bolt_cpu::CpuBackend;
use bolt_serialize::{
    inspect_tensor_set, load_tensor_set, save_tensor_set, tensor, Error, ErrorMode,
    TensorMeta, TensorRole, TensorSetLoadOptions, TensorSetSaveOptions, TensorToSave,
};
use bolt_tensor::Tensor;
use tempfile::TempDir;

type B = CpuBackend;
type D = f32;

fn make_tensor_to_save<'a>(
    name: &str,
    dtype: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
) -> TensorToSave<'a> {
    TensorToSave::new(
        TensorMeta::new(name, dtype, bolt_core::shape::Shape::from_slice(&shape).unwrap()).with_role(TensorRole::User),
        data,
    )
}

#[test]
fn tensor_set_roundtrip_small() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test_set");

    let w_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![0.5, 0.5];

    let w_bytes: Vec<u8> = bytemuck::cast_slice(&w_data).to_vec();
    let b_bytes: Vec<u8> = bytemuck::cast_slice(&b_data).to_vec();

    let tensors = vec![
        make_tensor_to_save("layer1.weight", DType::F32, vec![2, 2], w_bytes),
        make_tensor_to_save("layer1.bias", DType::F32, vec![2], b_bytes),
    ];

    save_tensor_set(
        tensors,
        &out_dir,
        &TensorSetSaveOptions {
            checksum: true,
            overwrite: false,
            ..Default::default()
        },
    )?;

    let set = load_tensor_set(
        &out_dir,
        &TensorSetLoadOptions {
            lazy: false,
            error_mode: ErrorMode::Strict,
        },
    )?;

    let w_view = set.get("layer1.weight")?;
    assert_eq!(w_view.dtype, DType::F32);
    assert_eq!(w_view.shape.as_slice(), &[2, 2]);
    let loaded_w: &[f32] = bytemuck::cast_slice(w_view.data);
    assert_eq!(loaded_w, &[1.0, 2.0, 3.0, 4.0]);

    let b_view = set.get("layer1.bias")?;
    assert_eq!(b_view.dtype, DType::F32);
    assert_eq!(b_view.shape.as_slice(), &[2]);
    let loaded_b: &[f32] = bytemuck::cast_slice(b_view.data);
    assert_eq!(loaded_b, &[0.5, 0.5]);

    Ok(())
}

#[test]
fn tensor_set_lazy_get_and_materialize() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("lazy_test");

    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    save_tensor_set(
        [make_tensor_to_save("test", DType::F32, vec![3], bytes.clone())],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    let set = load_tensor_set(
        &out_dir,
        &TensorSetLoadOptions {
            lazy: true,
            error_mode: ErrorMode::Strict,
        },
    )?;

    let view = set.get("test")?;
    assert_eq!(view.shape.as_slice(), &[3]);

    let materialized = set.materialize("test")?;
    assert_eq!(materialized, bytes);

    Ok(())
}

#[test]
fn tensor_set_checksum_strict() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("checksum_strict");

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    save_tensor_set(
        [make_tensor_to_save("test", DType::F32, vec![4], bytes)],
        &out_dir,
        &TensorSetSaveOptions {
            checksum: true,
            ..Default::default()
        },
    )?;

    let shard_path = out_dir.join("shards/weights-00001-of-00001.safetensors");
    let mut shard_data = fs::read(&shard_path).unwrap();
    if !shard_data.is_empty() {
        let mid = shard_data.len() / 2;
        shard_data[mid] ^= 0xFF;
        fs::write(&shard_path, shard_data).unwrap();
    }

    let result = load_tensor_set(
        &out_dir,
        &TensorSetLoadOptions {
            lazy: false,
            error_mode: ErrorMode::Strict,
        },
    );

    assert!(result.is_err());
    match result {
        Err(Error::ShardChecksumMismatch { shard, .. }) => {
            assert!(shard.to_string_lossy().contains("weights-00001-of-00001"));
        }
        Err(e) => panic!("Expected ShardChecksumMismatch, got {:?}", e),
        Ok(_) => panic!("Expected error but load succeeded"),
    }

    Ok(())
}

#[test]
fn tensor_set_checksum_permissive() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("checksum_permissive");

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    save_tensor_set(
        [make_tensor_to_save("test", DType::F32, vec![4], bytes)],
        &out_dir,
        &TensorSetSaveOptions {
            checksum: true,
            ..Default::default()
        },
    )?;

    let shard_path = out_dir.join("shards/weights-00001-of-00001.safetensors");
    let mut shard_data = fs::read(&shard_path).unwrap();
    if !shard_data.is_empty() {
        let mid = shard_data.len() / 2;
        shard_data[mid] ^= 0xFF;
        fs::write(&shard_path, shard_data).unwrap();
    }

    let set = load_tensor_set(
        &out_dir,
        &TensorSetLoadOptions {
            lazy: false,
            error_mode: ErrorMode::Permissive,
        },
    )?;

    let metas = set.list();
    assert_eq!(metas.len(), 1);

    let result = set.get("test");
    assert!(result.is_err());

    Ok(())
}

#[test]
fn tensor_set_inspect_without_loading() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("inspect_test");

    let data: Vec<f32> = vec![1.0; 1024];
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    save_tensor_set(
        [
            make_tensor_to_save("big_tensor", DType::F32, vec![32, 32], bytes.clone()),
            TensorToSave::new(
                TensorMeta::new("small", DType::F32, bolt_core::shape::Shape::from_slice(&[4]).unwrap())
                    .with_role(TensorRole::ModelParam),
                bytemuck::cast_slice::<f32, u8>(&[1.0, 2.0, 3.0, 4.0]).to_vec(),
            ),
        ],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    let metas = inspect_tensor_set(&out_dir)?;
    assert_eq!(metas.len(), 2);

    let big_meta = metas.iter().find(|m| m.name == "big_tensor").unwrap();
    assert_eq!(big_meta.shape.as_slice(), &[32, 32]);
    assert_eq!(big_meta.nbytes(), Some(4096)); // 1024 floats * 4 bytes

    let small_meta = metas.iter().find(|m| m.name == "small").unwrap();
    assert_eq!(small_meta.role, TensorRole::ModelParam);

    Ok(())
}

#[test]
fn overwrite_behavior() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("overwrite_test");

    let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&[1.0, 2.0]).to_vec();

    save_tensor_set(
        [make_tensor_to_save("v1", DType::F32, vec![2], bytes.clone())],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    let result = save_tensor_set(
        [make_tensor_to_save("v2", DType::F32, vec![2], bytes.clone())],
        &out_dir,
        &TensorSetSaveOptions {
            overwrite: false,
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(Error::DirectoryExists { .. })));

    save_tensor_set(
        [make_tensor_to_save("v2", DType::F32, vec![2], bytes)],
        &out_dir,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let set = load_tensor_set(&out_dir, &TensorSetLoadOptions::default())?;
    assert!(set.get("v2").is_ok());
    assert!(set.get("v1").is_err());

    Ok(())
}

#[test]
fn bolt_tensor_convenience_roundtrip() -> bolt_serialize::Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("tensor_convenience");

    let original = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .map_err(|e| Error::Safetensors {
            shard: out_dir.clone(),
            reason: e.to_string(),
        })?;

    tensor::save(
        "my_tensor",
        &original,
        &out_dir,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let loaded: Tensor<B, D> = tensor::load(
        "my_tensor",
        &out_dir,
        &backend,
        &TensorSetLoadOptions::default(),
    )?;

    assert_eq!(loaded.shape().as_slice(), &[2, 3]);

    let orig_data = original.to_vec().unwrap();
    let loaded_data = loaded.to_vec().unwrap();
    assert_eq!(orig_data, loaded_data);

    Ok(())
}

#[test]
fn multi_dtype_tensor_set() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("multi_dtype");

    let f32_data: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&[1.0, 2.0]).to_vec();
    let f64_data: Vec<u8> = bytemuck::cast_slice::<f64, u8>(&[3.0, 4.0]).to_vec();
    let i32_data: Vec<u8> = bytemuck::cast_slice::<i32, u8>(&[5, 6]).to_vec();

    save_tensor_set(
        [
            make_tensor_to_save("f32_tensor", DType::F32, vec![2], f32_data),
            make_tensor_to_save("f64_tensor", DType::F64, vec![2], f64_data),
            make_tensor_to_save("i32_tensor", DType::I32, vec![2], i32_data),
        ],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    let set = load_tensor_set(&out_dir, &TensorSetLoadOptions::default())?;

    assert_eq!(set.get("f32_tensor")?.dtype, DType::F32);
    assert_eq!(set.get("f64_tensor")?.dtype, DType::F64);
    assert_eq!(set.get("i32_tensor")?.dtype, DType::I32);

    Ok(())
}

#[test]
fn atomic_save_cleanup_on_failure() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("atomic_test");

    let invalid_tensor = TensorToSave {
        meta: TensorMeta::new("bad", DType::F32, bolt_core::shape::Shape::from_slice(&[100]).unwrap()), // expects 400 bytes
        data: vec![0u8; 10].into(),                          // only 10 bytes
    };

    let result = save_tensor_set([invalid_tensor], &out_dir, &TensorSetSaveOptions::default());

    assert!(result.is_err());
    assert!(!out_dir.exists());

    let entries: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
    assert!(
        entries.is_empty(),
        "Temp directories should be cleaned up on failure"
    );
}

#[test]
fn missing_tensor_error() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("missing_test");

    let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&[1.0]).to_vec();
    save_tensor_set(
        [make_tensor_to_save("exists", DType::F32, vec![1], bytes)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    let set = load_tensor_set(&out_dir, &TensorSetLoadOptions::default())?;

    let result = set.get("does_not_exist");
    assert!(matches!(result, Err(Error::TensorNotFound { .. })));

    Ok(())
}

#[test]
fn sharding_creates_multiple_files() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("sharding_test");

    let large_data: Vec<u8> = vec![0u8; 1000];

    save_tensor_set(
        [
            make_tensor_to_save("t1", DType::U8, vec![1000], large_data.clone()),
            make_tensor_to_save("t2", DType::U8, vec![1000], large_data.clone()),
            make_tensor_to_save("t3", DType::U8, vec![1000], large_data),
        ],
        &out_dir,
        &TensorSetSaveOptions {
            shard_max_bytes: Some(1500),
            ..Default::default()
        },
    )?;

    let shards_dir = out_dir.join("shards");
    let shard_count = fs::read_dir(&shards_dir).unwrap().count();
    assert!(shard_count >= 2, "Expected multiple shards, got {}", shard_count);

    let set = load_tensor_set(&out_dir, &TensorSetLoadOptions::default())?;
    assert!(set.get("t1").is_ok());
    assert!(set.get("t2").is_ok());
    assert!(set.get("t3").is_ok());

    Ok(())
}
