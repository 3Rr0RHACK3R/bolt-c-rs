use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_serialize::{
    CheckpointMeta, LoadOpts, Record, Role, SaveOpts, TensorFromCheckpoint, TensorToRecord,
    load_checkpoint, save_checkpoint,
};
use bolt_tensor::Tensor;

type B = CpuBackend;

#[test]
fn roundtrip_single_tensor_via_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_tensor");

    let original = Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let record: Record<'static> = original.to_record("test", Role::User)?;
    save_checkpoint(
        std::iter::once(Ok(record)),
        &out_path,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_path, &LoadOpts::default())?;
    let loaded: Tensor<B, f32> =
        Tensor::<B, f32>::restore_from_checkpoint(&ckpt, "test", &backend)?;

    assert_eq!(loaded.shape().as_slice(), &[2, 2]);
    assert_eq!(original.to_vec()?, loaded.to_vec()?);

    Ok(())
}

#[test]
fn roundtrip_multiple_tensors_via_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_many");

    let t1 = Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?;
    let t2 = Tensor::<B, f32>::from_slice(&backend, &[3.0, 4.0, 5.0], &[3])?;

    let r1 = t1.to_record("tensor1", Role::User)?;
    let r2 = t2.to_record("tensor2", Role::User)?;

    save_checkpoint(
        [Ok(r1), Ok(r2)],
        &out_path,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_path, &LoadOpts::default())?;
    let loaded1: Tensor<B, f32> =
        Tensor::<B, f32>::restore_from_checkpoint(&ckpt, "tensor1", &backend)?;
    let loaded2: Tensor<B, f32> =
        Tensor::<B, f32>::restore_from_checkpoint(&ckpt, "tensor2", &backend)?;

    assert_eq!(loaded1.shape().as_slice(), &[2]);
    assert_eq!(loaded2.shape().as_slice(), &[3]);
    assert_eq!(t1.to_vec()?, loaded1.to_vec()?);
    assert_eq!(t2.to_vec()?, loaded2.to_vec()?);

    Ok(())
}

#[test]
fn dtype_mismatch_error_on_tensor_load() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());
    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join("test_dtype");

    let original = Tensor::<B, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?;

    save_checkpoint(
        std::iter::once(Ok(original.to_record("test", Role::User)?)),
        &out_path,
        &CheckpointMeta::default(),
        &SaveOpts {
            overwrite: true,
            ..Default::default()
        },
    )?;

    let ckpt = load_checkpoint(&out_path, &LoadOpts::default())?;

    let result: bolt_serialize::Result<Tensor<B, f64>> =
        Tensor::<B, f64>::restore_from_checkpoint(&ckpt, "test", &backend);
    match result {
        Err(bolt_serialize::Error::DTypeMismatch {
            expected, found, ..
        }) => {
            assert_eq!(expected, bolt_core::DType::F64);
            assert_eq!(found, bolt_core::DType::F32);
            Ok(())
        }
        _ => panic!("expected DTypeMismatch"),
    }
}
