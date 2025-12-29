use std::fs;
use std::path::Path;

use crate::io::{
    build_tensor_entries, build_tensor_index, ensure_directory_structure,
    load_shards_with_verification, read_and_parse_manifest, validate_load_directory,
    validate_schema_version, write_manifest, write_shards_with_checksums,
};
use crate::manifest::{
    CHECKPOINT_MANIFEST_NAME, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest,
    CheckpointMetadataJson, ShardInfo,
};
use crate::tensor_set::TensorSetSaveOptions;
use crate::utils::create_temp_dir;
use crate::validation::{validate_no_duplicates, validate_tensor_bytes, validate_tensor_name};
use crate::{
    Error, ErrorMode, Result, TensorMeta, TensorRole, TensorSet, TensorToSave, TensorView,
};

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
    ensure_directory_structure(dir)?;

    let (plan, shard_checksums) = write_shards_with_checksums(
        tensors,
        dir,
        opts.tensor_set.shard_max_bytes,
        opts.tensor_set.alignment_bytes,
        opts.tensor_set.checksum,
    )?;

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
    build_tensor_entries(
        &mut manifest.tensors,
        tensors,
        &plan,
        opts.tensor_set.checksum,
    )?;

    let manifest_path = dir.join(CHECKPOINT_MANIFEST_NAME);
    write_manifest(&manifest, &manifest_path)?;

    Ok(())
}

pub fn load_checkpoint(dir: &Path, opts: &CheckpointLoadOptions) -> Result<Checkpoint> {
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
        opts.error_mode.clone(),
    )?;

    let index = build_tensor_index(
        &manifest.tensors,
        &manifest.shards.files,
        dir,
        &corrupted_shards,
    )?;

    let metadata = CheckpointMetadata {
        epoch: manifest.metadata.epoch,
        global_step: manifest.metadata.global_step,
        model_name: manifest.metadata.model_name.clone(),
        created_at_rfc3339: manifest.created_at.clone(),
        user: manifest.metadata.user.clone(),
    };

    Ok(Checkpoint {
        metadata,
        tensors: TensorSet::new(dir.to_path_buf(), shards, index),
    })
}

pub fn inspect_checkpoint(dir: &Path) -> Result<CheckpointMetadata> {
    let manifest_path = validate_load_directory(dir, CHECKPOINT_MANIFEST_NAME)?;

    let manifest = read_and_parse_manifest::<CheckpointManifest>(&manifest_path)?;

    Ok(CheckpointMetadata {
        epoch: manifest.metadata.epoch,
        global_step: manifest.metadata.global_step,
        model_name: manifest.metadata.model_name,
        created_at_rfc3339: manifest.created_at,
        user: manifest.metadata.user,
    })
}
