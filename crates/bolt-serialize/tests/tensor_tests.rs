use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_serialize::{Error, Result, TensorSetLoadOptions, TensorSetSaveOptions};
use bolt_tensor::Tensor;

type B = CpuBackend;

#[test]
fn roundtrip_single_tensor() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_tensor");

    let original =
        Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).map_err(|e| {
            Error::Safetensors {
                shard: out_path.clone(),
                reason: e.to_string(),
            }
        })?;

    bolt_serialize::tensor::save(
        "test",
        &original,
        &out_path,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let loaded: Tensor<B, f32> = bolt_serialize::tensor::load(
        "test",
        &out_path,
        &backend,
        &TensorSetLoadOptions::default(),
    )?;

    assert_eq!(loaded.shape().as_slice(), &[2, 2]);
    let original_data = original.to_vec().unwrap();
    let loaded_data = loaded.to_vec().unwrap();
    assert_eq!(original_data, loaded_data);

    Ok(())
}

#[test]
fn roundtrip_multiple_tensors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_many");

    let t1 = Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0], &[2]).map_err(|e| {
        Error::Safetensors {
            shard: out_path.clone(),
            reason: e.to_string(),
        }
    })?;
    let t2 = Tensor::<B, f32>::from_slice(&backend, &[3.0, 4.0, 5.0], &[3]).map_err(|e| {
        Error::Safetensors {
            shard: out_path.clone(),
            reason: e.to_string(),
        }
    })?;

    bolt_serialize::tensor::save_many(
        [("tensor1", &t1), ("tensor2", &t2)],
        &out_path,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let set = bolt_serialize::tensor::load_set(&out_path, &TensorSetLoadOptions::default())?;
    let loaded1: Tensor<B, f32> = bolt_serialize::tensor::load_from_set("tensor1", &set, &backend)?;
    let loaded2: Tensor<B, f32> = bolt_serialize::tensor::load_from_set("tensor2", &set, &backend)?;

    assert_eq!(loaded1.shape().as_slice(), &[2]);
    assert_eq!(loaded2.shape().as_slice(), &[3]);

    Ok(())
}

#[test]
fn dtype_mismatch_error() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_dtype");

    let original = Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0], &[2]).map_err(|e| {
        Error::Safetensors {
            shard: out_path.clone(),
            reason: e.to_string(),
        }
    })?;

    bolt_serialize::tensor::save(
        "test",
        &original,
        &out_path,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let result: Result<Tensor<B, f64>> = bolt_serialize::tensor::load(
        "test",
        &out_path,
        &backend,
        &TensorSetLoadOptions::default(),
    );

    assert!(result.is_err());
    if let Err(Error::DTypeMismatch {
        expected, found, ..
    }) = result
    {
        assert_eq!(expected, bolt_core::DType::F64);
        assert_eq!(found, bolt_core::DType::F32);
    } else {
        panic!("Expected DTypeMismatch error");
    }

    Ok(())
}
