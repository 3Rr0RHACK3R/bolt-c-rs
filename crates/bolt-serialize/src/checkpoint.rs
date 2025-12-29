use std::fs;
use std::path::Path;

use glob::Pattern;

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

/// User-provided metadata for a checkpoint.
///
/// This contains training-related metadata that the user controls.
/// System-generated fields like `written_at` are stored separately in [`CheckpointInfo`].
#[derive(Clone, Debug, Default)]
pub struct CheckpointMetadata {
    pub epoch: Option<u64>,
    pub global_step: Option<u64>,
    pub model_name: Option<String>,
    pub user: serde_json::Value,
}

/// Complete checkpoint information including both user metadata and system fields.
///
/// Returned when loading or inspecting a checkpoint.
#[derive(Clone, Debug)]
pub struct CheckpointInfo {
    /// User-provided metadata (epoch, global_step, model_name, user data).
    pub metadata: CheckpointMetadata,
    /// RFC 3339 timestamp of when the checkpoint was written to disk.
    /// This is system-generated and cannot be overridden by the user.
    pub written_at: String,
}

#[derive(Clone, Debug)]
pub struct CheckpointSaveOptions {
    pub include_optimizer: bool,
    pub include_rng: bool,
    pub include_buffers: bool,
    /// Patterns to exclude from the checkpoint. Uses glob pattern syntax.
    ///
    /// Common examples:
    /// - `"opt.*"` - exclude all optimizer tensors (e.g., `opt.weight.exp_avg`)
    /// - `"**/*.tmp*"` - exclude temporary buffers with `.tmp` in the name
    /// - `"decoder.*"` - exclude all tensors starting with `decoder.`
    ///
    /// Patterns are matched against the full tensor name. For glob syntax details,
    /// see the [glob crate documentation](https://docs.rs/glob/latest/glob/struct.Pattern.html).
    pub exclude: Vec<String>,
    pub tensor_set: TensorSetSaveOptions,
}

impl Default for CheckpointSaveOptions {
    fn default() -> Self {
        Self {
            include_optimizer: true,
            include_rng: true,
            include_buffers: true,
            exclude: Vec::new(),
            tensor_set: TensorSetSaveOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CheckpointLoadOptions {
    pub error_mode: ErrorMode,
    pub lazy: bool,
}

pub struct Checkpoint {
    pub info: CheckpointInfo,
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

pub fn save_checkpoint<'a, I>(
    tensors: I,
    dir: &Path,
    metadata: &CheckpointMetadata,
    opts: &CheckpointSaveOptions,
) -> Result<()>
where
    I: IntoIterator<Item = TensorToSave<'a>>,
{
    let tensors: Vec<TensorToSave<'a>> = tensors.into_iter().collect();

    // Compile exclude patterns upfront to validate them early
    let compiled_patterns: Result<Vec<Pattern>> = opts
        .exclude
        .iter()
        .map(|pattern| {
            Pattern::new(pattern).map_err(|e| Error::InvalidExcludePattern {
                pattern: pattern.clone(),
                source: e,
            })
        })
        .collect();
    let compiled_patterns = compiled_patterns?;

    let filtered: Vec<TensorToSave<'a>> = tensors
        .into_iter()
        .filter(|t| should_include_tensor(t, opts, &compiled_patterns))
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

    match write_checkpoint_to_dir(&filtered, &temp_dir, metadata, opts) {
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

fn should_include_tensor(
    tensor: &TensorToSave<'_>,
    opts: &CheckpointSaveOptions,
    compiled_patterns: &[Pattern],
) -> bool {
    match &tensor.meta.role {
        TensorRole::OptimizerState if !opts.include_optimizer => return false,
        TensorRole::RngState if !opts.include_rng => return false,
        TensorRole::ModelBuffer if !opts.include_buffers => return false,
        _ => {}
    }

    for pattern in compiled_patterns {
        if pattern.matches(&tensor.meta.name) {
            return false;
        }
    }

    true
}

fn write_checkpoint_to_dir<'a>(
    tensors: &[TensorToSave<'a>],
    dir: &Path,
    metadata: &CheckpointMetadata,
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
        epoch: metadata.epoch,
        global_step: metadata.global_step,
        model_name: metadata.model_name.clone(),
        user: metadata.user.clone(),
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

    let info = CheckpointInfo {
        metadata: CheckpointMetadata {
            epoch: manifest.metadata.epoch,
            global_step: manifest.metadata.global_step,
            model_name: manifest.metadata.model_name.clone(),
            user: manifest.metadata.user.clone(),
        },
        written_at: manifest.written_at.clone(),
    };

    Ok(Checkpoint {
        info,
        tensors: TensorSet::new(dir.to_path_buf(), shards, index),
    })
}

pub fn inspect_checkpoint(dir: &Path) -> Result<CheckpointInfo> {
    let manifest_path = validate_load_directory(dir, CHECKPOINT_MANIFEST_NAME)?;

    let manifest = read_and_parse_manifest::<CheckpointManifest>(&manifest_path)?;

    validate_schema_version(
        &manifest.schema_version,
        CHECKPOINT_SCHEMA_VERSION,
        manifest_path,
    )?;

    Ok(CheckpointInfo {
        metadata: CheckpointMetadata {
            epoch: manifest.metadata.epoch,
            global_step: manifest.metadata.global_step,
            model_name: manifest.metadata.model_name,
            user: manifest.metadata.user,
        },
        written_at: manifest.written_at,
    })
}
