use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::io::{
    build_tensor_entries, build_tensor_index, ensure_directory_structure,
    load_shards_with_verification, read_and_parse_manifest, validate_load_directory,
    validate_schema_version, write_manifest, write_shards_with_checksums,
};
use crate::manifest::{
    ShardInfo, TENSOR_SET_MANIFEST_NAME, TENSOR_SET_SCHEMA_VERSION, TensorSetManifest,
};
use crate::shard::LoadedShard;
use crate::utils::create_temp_dir;
use crate::validation::{validate_no_duplicates, validate_tensor_bytes, validate_tensor_name};
use crate::{Error, ErrorMode, Result, TensorMeta, TensorToSave, TensorView};

#[derive(Clone, Debug)]
pub struct TensorSetSaveOptions {
    pub shard_max_bytes: Option<u64>,
    pub alignment_bytes: u64,
    pub checksum: bool,
    pub overwrite: bool,
}

impl Default for TensorSetSaveOptions {
    fn default() -> Self {
        Self {
            shard_max_bytes: Some(2 * 1024 * 1024 * 1024), // 2 GiB
            alignment_bytes: 4096,
            checksum: true,
            overwrite: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorSetLoadOptions {
    pub lazy: bool,
    pub error_mode: ErrorMode,
}

impl Default for TensorSetLoadOptions {
    fn default() -> Self {
        Self {
            lazy: false,
            error_mode: ErrorMode::Strict,
        }
    }
}

pub(crate) struct TensorIndex {
    pub meta: TensorMeta,
    pub shard_idx: usize,
    pub key: String,
    #[allow(dead_code)]
    pub checksum: Option<String>,
    pub corrupted: bool,
}

pub struct TensorSet {
    artifact_dir: PathBuf,
    shards: Vec<LoadedShard>,
    index: HashMap<String, TensorIndex>,
}

impl TensorSet {
    pub(crate) fn new(
        artifact_dir: PathBuf,
        shards: Vec<LoadedShard>,
        index: HashMap<String, TensorIndex>,
    ) -> Self {
        Self {
            artifact_dir,
            shards,
            index,
        }
    }

    pub fn list(&self) -> Vec<TensorMeta> {
        self.index.values().map(|idx| idx.meta.clone()).collect()
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>> {
        let idx = self.index.get(name).ok_or_else(|| Error::TensorNotFound {
            name: name.to_string(),
            dir: self.artifact_dir.clone(),
        })?;

        if idx.corrupted {
            return Err(Error::TensorUnavailable {
                name: name.to_string(),
            });
        }

        let shard = &self.shards[idx.shard_idx];
        let tensors = shard.tensors()?;
        let tensor_view = tensors.tensor(&idx.key).map_err(|e| Error::Safetensors {
            shard: shard.path.clone(),
            reason: e.to_string(),
        })?;

        TensorView::from_safetensors_view(&tensor_view, &shard.path)
    }

    pub fn materialize(&self, name: &str) -> Result<Vec<u8>> {
        let view = self.get(name)?;
        Ok(view.data.to_vec())
    }

    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }
}

pub fn save_tensor_set<'a, I>(tensors: I, out_dir: &Path, opts: &TensorSetSaveOptions) -> Result<()>
where
    I: IntoIterator<Item = TensorToSave<'a>>,
{
    let tensors: Vec<TensorToSave<'a>> = tensors.into_iter().collect();

    for t in &tensors {
        validate_tensor_name(&t.meta.name)?;
        validate_tensor_bytes(t)?;
    }
    validate_no_duplicates(&tensors)?;

    if out_dir.exists() {
        if opts.overwrite {
            fs::remove_dir_all(out_dir).map_err(|e| Error::io(out_dir, e))?;
        } else {
            return Err(Error::DirectoryExists {
                dir: out_dir.to_path_buf(),
            });
        }
    }

    let temp_dir = create_temp_dir(out_dir, "artifact")?;

    match write_tensor_set_to_dir(&tensors, &temp_dir, opts) {
        Ok(()) => {
            fs::rename(&temp_dir, out_dir).map_err(|e| Error::AtomicRenameFailed {
                temp: temp_dir.clone(),
                final_dir: out_dir.to_path_buf(),
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

fn write_tensor_set_to_dir<'a>(
    tensors: &[TensorToSave<'a>],
    dir: &Path,
    opts: &TensorSetSaveOptions,
) -> Result<()> {
    ensure_directory_structure(dir)?;

    let (plan, shard_checksums) = write_shards_with_checksums(
        tensors,
        dir,
        opts.shard_max_bytes,
        opts.alignment_bytes,
        opts.checksum,
    )?;

    let mut manifest = TensorSetManifest::new();
    manifest.shards = ShardInfo {
        files: plan.shards.iter().map(|s| s.relative_path()).collect(),
        checksums: shard_checksums,
    };
    build_tensor_entries(&mut manifest.tensors, tensors, &plan, opts.checksum)?;

    let manifest_path = dir.join(TENSOR_SET_MANIFEST_NAME);
    write_manifest(&manifest, &manifest_path)?;

    if let Ok(file) = File::open(&manifest_path) {
        let _ = file.sync_all();
    }

    Ok(())
}

pub fn load_tensor_set(dir: &Path, opts: &TensorSetLoadOptions) -> Result<TensorSet> {
    let manifest_path = validate_load_directory(dir, TENSOR_SET_MANIFEST_NAME)?;

    let manifest = read_and_parse_manifest::<TensorSetManifest>(&manifest_path)?;

    validate_schema_version(
        &manifest.schema_version,
        TENSOR_SET_SCHEMA_VERSION,
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

    Ok(TensorSet::new(dir.to_path_buf(), shards, index))
}

pub fn inspect_tensor_set(dir: &Path) -> Result<Vec<TensorMeta>> {
    let manifest_path = validate_load_directory(dir, TENSOR_SET_MANIFEST_NAME)?;

    let manifest = read_and_parse_manifest::<TensorSetManifest>(&manifest_path)?;

    let mut metas = Vec::new();
    for (name, entry) in &manifest.tensors {
        let dtype = entry.parse_dtype().ok_or_else(|| Error::Safetensors {
            shard: dir.to_path_buf(),
            reason: format!("unknown dtype '{}' for tensor '{}'", entry.dtype, name),
        })?;

        metas.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape: entry.shape.clone(),
            role: entry.role.clone(),
            group: entry.group,
        });
    }

    Ok(metas)
}
