use bolt_core::shape::Shape;
use bolt_core::DType;
use bolt_serialize::{TensorMeta, TensorRole, TensorToSave};

fn make_tensor(name: &str, size: usize) -> TensorToSave<'static> {
    TensorToSave {
        meta: TensorMeta {
            name: name.to_string(),
            dtype: DType::F32,
            shape: Shape::from_slice(&[size / 4]).unwrap(),
            role: TensorRole::User,
            group: 0,
        },
        data: vec![0u8; size].into(),
    }
}

#[test]
fn shard_integration_via_save() {
    use bolt_serialize::{
        load_tensor_set, save_tensor_set, TensorSetLoadOptions, TensorSetSaveOptions,
    };
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test_sharding");

    let tensors = vec![
        make_tensor("c_tensor", 100),
        make_tensor("a_tensor", 100),
        make_tensor("b_tensor", 100),
    ];

    let result = save_tensor_set(
        tensors,
        &out_dir,
        &TensorSetSaveOptions {
            overwrite: true,
            ..Default::default()
        },
    );

    assert!(result.is_ok());

    let set = load_tensor_set(&out_dir, &TensorSetLoadOptions::default()).unwrap();

    assert!(set.get("a_tensor").is_ok());
    assert!(set.get("b_tensor").is_ok());
    assert!(set.get("c_tensor").is_ok());
}

#[test]
fn many_tensors_creates_shards() {
    use bolt_serialize::{save_tensor_set, TensorSetSaveOptions};
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test_many_shards");

    let tensors: Vec<_> = (0..10)
        .map(|i| make_tensor(&format!("tensor_{:03}", i), 1000))
        .collect();

    let result = save_tensor_set(
        tensors,
        &out_dir,
        &TensorSetSaveOptions {
            overwrite: true,
            shard_max_bytes: Some(2000),
            ..Default::default()
        },
    );

    assert!(result.is_ok());

    let shards_dir = out_dir.join("shards");
    assert!(shards_dir.exists());

    let shard_files: Vec<_> = fs::read_dir(&shards_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .collect();

    assert!(
        shard_files.len() > 1,
        "Expected multiple shards, got {}",
        shard_files.len()
    );
}
