use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::manifest::{dtype_to_name, TensorEntry, TensorLocation, SHARDS_DIR};
use crate::shard::{
    compute_bytes_checksum, compute_file_checksum, plan_shards, verify_checksum, write_shard,
    LoadedShard, ShardPlan,
};
use crate::validation::validate_shard_path;
use crate::{Error, ErrorMode, Result, TensorMeta, TensorToSave};


/// Ensures the directory structure exists (creates dir and shards subdirectory).
pub(crate) fn ensure_directory_structure(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let shards_dir = dir.join(SHARDS_DIR);
    fs::create_dir_all(&shards_dir).map_err(|e| Error::io(&shards_dir, e))?;

    Ok(())
}

/// Plans shards, writes them to disk, and computes checksums.
/// Returns the shard plan and checksums (empty if checksum is disabled).
pub(crate) fn write_shards_with_checksums<'a>(
    tensors: &[TensorToSave<'a>],
    dir: &Path,
    shard_max_bytes: Option<u64>,
    alignment_bytes: u64,
    compute_checksum: bool,
) -> Result<(ShardPlan, Vec<String>)> {
    let plan = plan_shards(tensors, shard_max_bytes)?;

    let mut shard_checksums = Vec::new();
    for shard in &plan.shards {
        let shard_path = dir.join(shard.relative_path());
        write_shard(shard, tensors, &shard_path, alignment_bytes)?;

        if compute_checksum {
            let checksum = compute_file_checksum(&shard_path)?;
            shard_checksums.push(checksum);
        }
    }

    Ok((plan, shard_checksums))
}

/// Builds tensor entries for the manifest from the shard plan.
pub(crate) fn build_tensor_entries<'a>(
    tensor_entries: &mut std::collections::BTreeMap<String, TensorEntry>,
    tensors: &[TensorToSave<'a>],
    plan: &ShardPlan,
    compute_checksum: bool,
) -> Result<()> {
    let tensor_map: HashMap<&str, &TensorToSave<'a>> =
        tensors.iter().map(|t| (t.meta.name.as_str(), t)).collect();

    for (shard_idx, shard) in plan.shards.iter().enumerate() {
        for name in &shard.tensor_names {
            let tensor = tensor_map[name.as_str()];
            let tensor_checksum = if compute_checksum {
                Some(compute_bytes_checksum(&tensor.data))
            } else {
                None
            };

            tensor_entries.insert(
                name.clone(),
                TensorEntry {
                    role: tensor.meta.role.clone(),
                    group: tensor.meta.group,
                    dtype: dtype_to_name(tensor.meta.dtype).to_string(),
                    shape: tensor.meta.shape.clone(),
                    location: TensorLocation {
                        shard: plan.shards[shard_idx].relative_path(),
                        key: name.clone(),
                    },
                    checksum: tensor_checksum,
                },
            );
        }
    }

    Ok(())
}

/// Writes a manifest to disk as pretty-printed JSON.
pub(crate) fn write_manifest(manifest: &impl Serialize, path: &Path) -> Result<()> {
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, manifest).map_err(|e| Error::ManifestParse {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;

    Ok(())
}

// ============================================================================
// Load Helpers
// ============================================================================

/// Validates that the directory exists and returns the manifest path.
pub(crate) fn validate_load_directory(dir: &Path, manifest_name: &str) -> Result<PathBuf> {
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

/// Reads and parses a manifest from JSON.
pub(crate) fn read_and_parse_manifest<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<T> {
    let manifest_content =
        fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let manifest: T = serde_json::from_str(&manifest_content).map_err(|e| Error::ManifestParse {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;

    Ok(manifest)
}

/// Validates that the manifest schema version matches the expected version.
pub(crate) fn validate_schema_version(
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

/// Loads shards with checksum verification.
/// Returns the loaded shards and a list of corrupted shard indices.
pub(crate) fn load_shards_with_verification(
    shard_files: &[String],
    shard_checksums: &[String],
    dir: &Path,
    lazy: bool,
    error_mode: ErrorMode,
) -> Result<(Vec<LoadedShard>, Vec<usize>)> {
    let mut shards = Vec::new();
    let mut corrupted_shards = Vec::new();

    for (i, shard_path_str) in shard_files.iter().enumerate() {
        validate_shard_path(shard_path_str, dir)?;
        let shard_path = dir.join(shard_path_str);

        if !shard_path.exists() {
            return Err(Error::ShardNotFound { path: shard_path });
        }

        // Verify checksum if available
        if i < shard_checksums.len() {
            let expected = &shard_checksums[i];
            match verify_checksum(&shard_path, expected) {
                Ok(()) => {}
                Err(e) => {
                    if matches!(error_mode, ErrorMode::Strict) {
                        return Err(e);
                    }
                    corrupted_shards.push(i);
                }
            }
        }

        // Load shard (eager or lazy)
        let shard = if lazy {
            LoadedShard::load_mmap(&shard_path)?
        } else {
            LoadedShard::load_eager(&shard_path)?
        };

        shards.push(shard);
    }

    Ok((shards, corrupted_shards))
}

/// Builds the tensor index and shapes map from the manifest.
/// Returns (index, shapes).
pub(crate) fn build_tensor_index(
    tensor_entries: &std::collections::BTreeMap<String, TensorEntry>,
    shard_files: &[String],
    dir: &Path,
    corrupted_shards: &[usize],
) -> Result<(HashMap<String, crate::tensor_set::TensorIndex>, HashMap<String, bolt_core::shape::Shape>)> {
    let shard_path_to_idx: HashMap<&str, usize> = shard_files
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();

    let mut index = HashMap::new();
    let mut shapes = HashMap::new();

    for (name, entry) in tensor_entries {
        let shard_idx = *shard_path_to_idx
            .get(entry.location.shard.as_str())
            .ok_or_else(|| Error::Safetensors {
                shard: dir.join(&entry.location.shard),
                reason: format!("shard '{}' not found in manifest", entry.location.shard),
            })?;

        let dtype = entry.parse_dtype().ok_or_else(|| Error::Safetensors {
            shard: dir.join(&entry.location.shard),
            reason: format!("unknown dtype '{}'", entry.dtype),
        })?;

        let corrupted = corrupted_shards.contains(&shard_idx);

        index.insert(
            name.clone(),
            crate::tensor_set::TensorIndex {
                meta: TensorMeta {
                    name: name.clone(),
                    dtype,
                    shape: entry.shape.clone(),
                    role: entry.role.clone(),
                    group: entry.group,
                },
                shard_idx,
                key: entry.location.key.clone(),
                checksum: entry.checksum.clone(),
                corrupted,
            },
        );
        shapes.insert(name.clone(), entry.shape.clone());
    }

    Ok((index, shapes))
}
