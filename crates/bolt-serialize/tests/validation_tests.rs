use std::fs;
use std::path::Path;

use bolt_core::{DType, shape::Shape};
use bolt_serialize::{
    CheckpointMeta, Error, LoadOpts, Record, RecordMeta, Role, SaveOpts, load_checkpoint,
    save_checkpoint,
};
use tempfile::TempDir;

fn make_record(name: &str, byte_len: usize) -> Record<'static> {
    debug_assert!(
        byte_len % 4 == 0,
        "record byte length must be divisible by 4 for F32 dtype"
    );
    Record::new(
        RecordMeta::new(
            name,
            DType::F32,
            Shape::from_slice(&[byte_len / 4]).unwrap(),
        )
        .with_role(Role::User),
        vec![0u8; byte_len],
    )
}

#[test]
fn reject_empty_record_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        std::iter::once(Ok(make_record("", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_nul_in_record_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        std::iter::once(Ok(make_record("foo\0bar", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_path_separator_in_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        std::iter::once(Ok(make_record("foo/bar", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));

    let result = save_checkpoint(
        std::iter::once(Ok(make_record("foo\\bar", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_reserved_directory_name() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        std::iter::once(Ok(make_record("..", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );
    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_hidden_file_prefix() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        std::iter::once(Ok(make_record(".hidden", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    assert!(matches!(result, Err(Error::InvalidName { .. })));
}

#[test]
fn reject_duplicate_record_names() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let result = save_checkpoint(
        [
            Ok(make_record("same_name", 100)),
            Ok(make_record("same_name", 100)),
        ],
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    assert!(matches!(result, Err(Error::DuplicateName { .. })));
}

#[test]
fn reject_byte_size_mismatch() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("test");

    let bad_record = Record {
        meta: RecordMeta::new("bad", DType::F32, Shape::from_slice(&[100]).unwrap()),
        data: vec![0u8; 50].into(),
    };

    let result = save_checkpoint(
        std::iter::once(Ok(bad_record)),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    match result {
        Err(Error::ByteSizeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 400);
            assert_eq!(actual, 50);
        }
        other => panic!("expected ByteSizeMismatch error, got: {other:?}"),
    }
}

#[test]
fn accept_valid_record_names() {
    let tmp = TempDir::new().unwrap();

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
        let result = save_checkpoint(
            std::iter::once(Ok(make_record(name, 100))),
            &new_dir,
            &CheckpointMeta::default(),
            &SaveOpts::default(),
        );
        assert!(result.is_ok(), "name '{name}' should be valid");
    }
}

#[test]
fn directory_not_found_on_load() {
    let result = load_checkpoint(
        Path::new("/nonexistent/path/that/does/not/exist"),
        &LoadOpts::default(),
    );

    assert!(matches!(result, Err(Error::DirectoryNotFound { .. })));
}

#[test]
fn manifest_not_found_on_load() {
    let tmp = TempDir::new().unwrap();
    let empty_dir = tmp.path().join("empty");
    fs::create_dir(&empty_dir).unwrap();

    let result = load_checkpoint(&empty_dir, &LoadOpts::default());

    assert!(matches!(result, Err(Error::ManifestNotFound { .. })));
}

#[test]
fn schema_version_mismatch() -> bolt_serialize::Result<()> {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("schema_test");

    save_checkpoint(
        std::iter::once(Ok(make_record("test", 100))),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    )?;

    let manifest_path = out_dir.join("bolt-checkpoint.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["schema_version"] = serde_json::Value::String("bolt-checkpoint:999".to_string());
    fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

    let result = load_checkpoint(&out_dir, &LoadOpts::default());

    match result {
        Err(Error::SchemaVersionMismatch {
            expected, found, ..
        }) => {
            assert_eq!(expected, "bolt-checkpoint:1");
            assert_eq!(found, "bolt-checkpoint:999");
        }
        _ => panic!("expected SchemaVersionMismatch error"),
    }

    Ok(())
}

#[test]
fn manifest_parse_error() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("bad_manifest");
    fs::create_dir_all(&dir).unwrap();

    fs::write(dir.join("bolt-checkpoint.json"), "{ invalid json }").unwrap();

    let result = load_checkpoint(&dir, &LoadOpts::default());

    assert!(matches!(result, Err(Error::ManifestParse { .. })));
}

#[test]
fn atomic_save_cleanup_on_failure() {
    let tmp = TempDir::new().unwrap();
    let out_dir = tmp.path().join("atomic_test");

    let bad_record = Record {
        meta: RecordMeta::new("bad", DType::F32, Shape::from_slice(&[100]).unwrap()),
        data: vec![0u8; 10].into(),
    };

    let result = save_checkpoint(
        std::iter::once(Ok(bad_record)),
        &out_dir,
        &CheckpointMeta::default(),
        &SaveOpts::default(),
    );

    assert!(result.is_err());
    assert!(!out_dir.exists(), "Output directory should not exist after failed save");

    let entries: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
    assert!(
        entries.is_empty(),
        "Temp directories should be cleaned up on failure, found: {:?}",
        entries
    );
}
