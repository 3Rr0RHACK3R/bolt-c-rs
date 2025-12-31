//! Checkpoint loading - start here for any load-related issues.
//!
//! Flow: validate directory -> load shards -> build index -> return Checkpoint

mod checkpoint;
mod index;
pub(crate) mod shard;

pub use checkpoint::{Checkpoint, CheckpointInfo, CheckpointMeta};

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::format::{
    CheckpointManifest, CHECKPOINT_MANIFEST_NAME, CHECKPOINT_SCHEMA_VERSION,
};
use crate::options::LoadOpts;
use crate::{Error, Result};

use index::build_record_index;
use shard::load_shards_with_verification;

pub fn load_checkpoint(dir: &Path, opts: &LoadOpts) -> Result<Checkpoint> {
    let manifest_path = validate_load_directory(dir, CHECKPOINT_MANIFEST_NAME)?;
    let manifest = read_and_parse_manifest::<CheckpointManifest>(&manifest_path)?;

    validate_schema_version(
        &manifest.schema_version,
        CHECKPOINT_SCHEMA_VERSION,
        manifest_path,
    )?;

    let (shards, corrupted_shards) = load_shards_with_verification(
        &manifest.shards.files,
        &manifest.shards.checksums,
        dir,
        opts.lazy,
        opts.on_error.clone(),
    )?;

    let index = build_record_index(
        &manifest.tensors,
        &manifest.shards.files,
        dir,
        &corrupted_shards,
    )?;

    let info = CheckpointInfo {
        meta: CheckpointMeta {
            epoch: manifest.metadata.epoch,
            global_step: manifest.metadata.global_step,
            model_name: manifest.metadata.model_name.clone(),
            user: manifest.metadata.user.clone(),
        },
        written_at: manifest.written_at.clone(),
    };

    Ok(Checkpoint {
        info,
        dir: dir.to_path_buf(),
        shards,
        index,
    })
}

pub fn inspect(dir: &Path) -> Result<CheckpointInfo> {
    let manifest_path = validate_load_directory(dir, CHECKPOINT_MANIFEST_NAME)?;
    let manifest = read_and_parse_manifest::<CheckpointManifest>(&manifest_path)?;

    validate_schema_version(
        &manifest.schema_version,
        CHECKPOINT_SCHEMA_VERSION,
        manifest_path,
    )?;

    Ok(CheckpointInfo {
        meta: CheckpointMeta {
            epoch: manifest.metadata.epoch,
            global_step: manifest.metadata.global_step,
            model_name: manifest.metadata.model_name,
            user: manifest.metadata.user,
        },
        written_at: manifest.written_at,
    })
}

fn validate_load_directory(dir: &Path, manifest_name: &str) -> Result<PathBuf> {
    if !dir.exists() {
        return Err(Error::DirectoryNotFound {
            dir: dir.to_path_buf(),
        });
    }

    let manifest_path = dir.join(manifest_name);
    if !manifest_path.exists() {
        return Err(Error::ManifestNotFound {
            path: manifest_path,
        });
    }

    Ok(manifest_path)
}

fn read_and_parse_manifest<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let content = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    serde_json::from_str(&content).map_err(|e| Error::ManifestParse {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

fn validate_schema_version(
    manifest_version: &str,
    expected: &str,
    file: PathBuf,
) -> Result<()> {
    if manifest_version != expected {
        return Err(Error::SchemaVersionMismatch {
            file,
            expected: expected.to_string(),
            found: manifest_version.to_string(),
        });
    }

    Ok(())
}
