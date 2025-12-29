use std::fs;
use std::path::Path;

use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    Error, TensorMeta, TensorRole, TensorSetSaveOptions, TensorToSave, save_tensor_set,
};
use tempfile::TempDir;

fn make_tensor<'a>(name: &str, byte_len: usize) -> TensorToSave<'a> {
    debug_assert!(
        byte_len % 4 == 0,
        "tensor byte length must be divisible by 4 for F32 dtype"
    );
    TensorToSave::new(
        TensorMeta::new(
            name,
            DType::F32,
            Shape::from_slice(&[byte_len / 4]).unwrap(),
        )
        .with_role(TensorRole::User),
        vec![0u8; byte_len],
    )
}

#[test]
fn reject_empty_tensor_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor("", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_nul_in_tensor_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor("foo\0bar", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_path_separator_in_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor("foo/bar", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));

    let result = save_tensor_set(
        [make_tensor("foo\\bar", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_reserved_directory_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor("..", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_hidden_file_prefix() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor(".hidden", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_duplicate_tensor_names() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_tensor_set(
        [make_tensor("same_name", 100), make_tensor("same_name", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    );

    assert!(matches!(result, Err(Error::DuplicateName { .. })));
}

#[test]
fn reject_byte_size_mismatch() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let bad_tensor = TensorToSave {
        meta: TensorMeta::new("bad", DType::F32, Shape::from_slice(&[100]).unwrap()), // expects 400 bytes
        data: vec![0u8; 50].into(), // only 50 bytes
    };

    let result = save_tensor_set([bad_tensor], &out_dir, &TensorSetSaveOptions::default());

    match result {
        Err(Error::ByteSizeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 400);
            assert_eq!(actual, 50);
        }
        _ => panic!("Expected ByteSizeMismatch error"),
    }
}

#[test]
fn accept_valid_tensor_names() {
    let tmp = TempDir::new().unwrap();

    // All these should be valid
    let valid_names = vec![
        "layer1.weight",
        "encoder_block_0_attention_query",
        "model.layers.0.self_attn.q_proj.weight",
        "simple",
        "a",
        "tensor_123",
        "foo..bar",
    ];

    for name in valid_names {
        let new_dir = tmp.path().join(format!("test_{}", name.replace('.', "_")));
        let result = save_tensor_set(
            [make_tensor(name, 100)],
            &new_dir,
            &TensorSetSaveOptions::default(),
        );
        assert!(result.is_ok(), "Name '{}' should be valid", name);
    }
}

#[test]
fn directory_not_found_on_load() {
    let result = bolt_serialize::load_tensor_set(
        Path::new("/nonexistent/path/that/does/not/exist"),
        &bolt_serialize::TensorSetLoadOptions::default(),
    );

    assert!(matches!(result, Err(Error::DirectoryNotFound { .. })));
}

#[test]
fn manifest_not_found_on_load() {
    let tmp = TempDir::new().unwrap();
    let empty_dir = tmp.path().join("empty");
    fs::create_dir(&empty_dir).unwrap();

    let result = bolt_serialize::load_tensor_set(
        &empty_dir,
        &bolt_serialize::TensorSetLoadOptions::default(),
    );

    assert!(matches!(result, Err(Error::ManifestNotFound { .. })));
}

#[test]
fn schema_version_mismatch() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("schema_test");

    // Save a valid tensor set
    save_tensor_set(
        [make_tensor("test", 100)],
        &out_dir,
        &TensorSetSaveOptions::default(),
    )?;

    // Corrupt the manifest with wrong schema version
    let manifest_path = out_dir.join("bolt-tensorset.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["schema_version"] = serde_json::Value::String("bolt-tensorset:999".to_string());
    fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

    let result =
        bolt_serialize::load_tensor_set(&out_dir, &bolt_serialize::TensorSetLoadOptions::default());

    match result {
        Err(Error::SchemaVersionMismatch {
            expected, found, ..
        }) => {
            assert_eq!(expected, "bolt-tensorset:1");
            assert_eq!(found, "bolt-tensorset:999");
        }
        _ => panic!("Expected SchemaVersionMismatch error"),
    }

    Ok(())
}

#[test]
fn manifest_parse_error() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("bad_manifest");
    fs::create_dir_all(&dir).unwrap();

    // Write invalid JSON
    fs::write(dir.join("bolt-tensorset.json"), "{ invalid json }").unwrap();

    let result =
        bolt_serialize::load_tensor_set(&dir, &bolt_serialize::TensorSetLoadOptions::default());

    assert!(matches!(result, Err(Error::ManifestParse { .. })));
}
