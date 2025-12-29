use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use crate::manifest::{
    dtype_to_name, CheckpointManifest, CheckpointMetadataJson, ShardInfo, TensorEntry,
    TensorLocation, CHECKPOINT_MANIFEST_NAME, CHECKPOINT_SCHEMA_VERSION, SHARDS_DIR,
};
use crate::shard::{
    compute_bytes_checksum, compute_file_checksum, plan_shards, verify_checksum, write_shard,
    LoadedShard,
};
use crate::tensor_set::{TensorIndex, TensorSetSaveOptions};
use crate::validation::{
    validate_no_duplicates, validate_shard_path, validate_tensor_bytes, validate_tensor_name,
};
use crate::utils::create_temp_dir;
use crate::{Error, ErrorMode, Result, TensorMeta, TensorRole, TensorSet, TensorToSave, TensorView};

#[derive(Clone, Debug, Default)]
pub struct CheckpointMetadata {
    pub epoch: Option<u64>,
    pub global_step: Option<u64>,
    pub model_name: Option<String>,
    pub created_at_rfc3339: String,
    pub user: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct CheckpointSaveOptions {
    pub include_optimizer: bool,
    pub include_rng: bool,
    pub include_buffers: bool,
    pub exclude: Vec<String>,
    pub tensor_set: TensorSetSaveOptions,
    pub metadata: CheckpointMetadata,
}

impl Default for CheckpointSaveOptions {
    fn default() -> Self {
        Self {
            include_optimizer: true,
            include_rng: true,
            include_buffers: true,
            exclude: Vec::new(),
            tensor_set: TensorSetSaveOptions::default(),
            metadata: CheckpointMetadata::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CheckpointLoadOptions {
    pub error_mode: ErrorMode,
    pub lazy: bool,
}

pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    pub tensors: TensorSet,
}

impl Checkpoint {
    pub fn list(&self) -> Vec<TensorMeta> {
        self.tensors.list()
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>> {
        self.tensors.get(name)
    }

    pub fn materialize(&self, name: &str) -> Result<Vec<u8>> {
        self.tensors.materialize(name)
    }
}

pub fn save_checkpoint<'a, I>(tensors: I, dir: &Path, opts: &CheckpointSaveOptions) -> Result<()>
where
    I: IntoIterator<Item = TensorToSave<'a>>,
{
    let tensors: Vec<TensorToSave<'a>> = tensors.into_iter().collect();

    // Filter tensors based on options
    let filtered: Vec<TensorToSave<'a>> = tensors
        .into_iter()
        .filter(|t| should_include_tensor(t, opts))
        .collect();

    for t in &filtered {
        validate_tensor_name(&t.meta.name)?;
        validate_tensor_bytes(t)?;
    }
    validate_no_duplicates(&filtered)?;

    if dir.exists() {
        if opts.tensor_set.overwrite {
            fs::remove_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        } else {
            return Err(Error::DirectoryExists {
                dir: dir.to_path_buf(),
            });
        }
    }

    let temp_dir = create_temp_dir(dir, "checkpoint")?;

    match write_checkpoint_to_dir(&filtered, &temp_dir, opts) {
        Ok(()) => {
            fs::rename(&temp_dir, dir).map_err(|e| Error::AtomicRenameFailed {
                temp: temp_dir.clone(),
                final_dir: dir.to_path_buf(),
                source: e,
            })?;
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&temp_dir);
            Err(e)
        }
    }
}

fn should_include_tensor(tensor: &TensorToSave<'_>, opts: &CheckpointSaveOptions) -> bool {
    match &tensor.meta.role {
        TensorRole::OptimizerState if !opts.include_optimizer => return false,
        TensorRole::RngState if !opts.include_rng => return false,
        TensorRole::ModelBuffer if !opts.include_buffers => return false,
        _ => {}
    }

    for pattern in &opts.exclude {
        if tensor.meta.name.contains(pattern) {
            return false;
        }
    }

    true
}

fn write_checkpoint_to_dir<'a>(
    tensors: &[TensorToSave<'a>],
    dir: &Path,
    opts: &CheckpointSaveOptions,
) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let shards_dir = dir.join(SHARDS_DIR);
    fs::create_dir_all(&shards_dir).map_err(|e| Error::io(&shards_dir, e))?;

    let plan = plan_shards(tensors, opts.tensor_set.shard_max_bytes)?;

    let mut shard_checksums = Vec::new();
    for shard in &plan.shards {
        let shard_path = dir.join(shard.relative_path());
        write_shard(shard, tensors, &shard_path, opts.tensor_set.alignment_bytes)?;

        if opts.tensor_set.checksum {
            let checksum = compute_file_checksum(&shard_path)?;
            shard_checksums.push(checksum);
        }
    }

    let mut manifest = CheckpointManifest::new();
    manifest.metadata = CheckpointMetadataJson {
        epoch: opts.metadata.epoch,
        global_step: opts.metadata.global_step,
        model_name: opts.metadata.model_name.clone(),
        user: opts.metadata.user.clone(),
    };
    manifest.shards = ShardInfo {
        files: plan.shards.iter().map(|s| s.relative_path()).collect(),
        checksums: shard_checksums,
    };

    let tensor_map: HashMap<&str, &TensorToSave<'a>> =
        tensors.iter().map(|t| (t.meta.name.as_str(), t)).collect();

    for (shard_idx, shard) in plan.shards.iter().enumerate() {
        for name in &shard.tensor_names {
            let tensor = tensor_map[name.as_str()];
            let tensor_checksum = if opts.tensor_set.checksum {
                Some(compute_bytes_checksum(&tensor.data))
            } else {
                None
            };

            manifest.tensors.insert(
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

    let manifest_path = dir.join(CHECKPOINT_MANIFEST_NAME);
    let file = File::create(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &manifest).map_err(|e| Error::ManifestParse {
        path: manifest_path.clone(),
        reason: e.to_string(),
    })?;

    Ok(())
}

pub fn load_checkpoint(dir: &Path, opts: &CheckpointLoadOptions) -> Result<Checkpoint> {
    if !dir.exists() {
        return Err(Error::DirectoryNotFound {
            dir: dir.to_path_buf(),
        });
    }

    let manifest_path = dir.join(CHECKPOINT_MANIFEST_NAME);
    if !manifest_path.exists() {
        return Err(Error::ManifestNotFound {
            path: manifest_path,
        });
    }

    let manifest_content =
        fs::read_to_string(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let manifest: CheckpointManifest =
        serde_json::from_str(&manifest_content).map_err(|e| Error::ManifestParse {
            path: manifest_path.clone(),
            reason: e.to_string(),
        })?;

    if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(Error::SchemaVersionMismatch {
            file: manifest_path,
            expected: CHECKPOINT_SCHEMA_VERSION.to_string(),
            found: manifest.schema_version.clone(),
        });
    }

    let mut shards = Vec::new();
    let mut corrupted_shards = Vec::new();

    for (i, shard_path_str) in manifest.shards.files.iter().enumerate() {
        validate_shard_path(shard_path_str, dir)?;
        let shard_path = dir.join(shard_path_str);

        if !shard_path.exists() {
            return Err(Error::ShardNotFound { path: shard_path });
        }

        if i < manifest.shards.checksums.len() {
            let expected = &manifest.shards.checksums[i];
            match verify_checksum(&shard_path, expected) {
                Ok(()) => {}
                Err(e) => {
                    if matches!(opts.error_mode, ErrorMode::Strict) {
                        return Err(e);
                    }
                    corrupted_shards.push(i);
                }
            }
        }

        let shard = if opts.lazy {
            LoadedShard::load_mmap(&shard_path)?
        } else {
            LoadedShard::load_eager(&shard_path)?
        };

        shards.push(shard);
    }

    let shard_path_to_idx: HashMap<&str, usize> = manifest
        .shards
        .files
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();

    let mut index = HashMap::new();
    let mut shapes = HashMap::new();

    for (name, entry) in &manifest.tensors {
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
            TensorIndex {
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

    let metadata = CheckpointMetadata {
        epoch: manifest.metadata.epoch,
        global_step: manifest.metadata.global_step,
        model_name: manifest.metadata.model_name.clone(),
        created_at_rfc3339: manifest.created_at.clone(),
        user: manifest.metadata.user.clone(),
    };

    Ok(Checkpoint {
        metadata,
        tensors: TensorSet::new(dir.to_path_buf(), shards, index, shapes),
    })
}

pub fn inspect_checkpoint(dir: &Path) -> Result<CheckpointMetadata> {
    if !dir.exists() {
        return Err(Error::DirectoryNotFound {
            dir: dir.to_path_buf(),
        });
    }

    let manifest_path = dir.join(CHECKPOINT_MANIFEST_NAME);
    if !manifest_path.exists() {
        return Err(Error::ManifestNotFound {
            path: manifest_path,
        });
    }

    let content = fs::read_to_string(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let manifest: CheckpointManifest =
        serde_json::from_str(&content).map_err(|e| Error::ManifestParse {
            path: manifest_path.clone(),
            reason: e.to_string(),
        })?;

    Ok(CheckpointMetadata {
        epoch: manifest.metadata.epoch,
        global_step: manifest.metadata.global_step,
        model_name: manifest.metadata.model_name,
        created_at_rfc3339: manifest.created_at,
        user: manifest.metadata.user,
    })
}


