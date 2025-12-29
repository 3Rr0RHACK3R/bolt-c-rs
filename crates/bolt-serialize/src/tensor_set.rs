use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::manifest::{
    dtype_to_name, ShardInfo, TensorEntry, TensorLocation, TensorSetManifest, SHARDS_DIR,
    TENSOR_SET_MANIFEST_NAME, TENSOR_SET_SCHEMA_VERSION,
};
use crate::shard::{
    compute_bytes_checksum, compute_file_checksum, dtype_from_safe, plan_shards, verify_checksum,
    write_shard, LoadedShard,
};
use crate::validation::{
    validate_no_duplicates, validate_shard_path, validate_tensor_bytes, validate_tensor_name,
};
use crate::utils::create_temp_dir;
use crate::{Error, ErrorMode, Result, TensorMeta, TensorToSave, TensorView};
use bolt_core::shape::Shape;

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
    shapes: HashMap<String, Shape>,
}

impl TensorSet {
    pub(crate) fn new(
        artifact_dir: PathBuf,
        shards: Vec<LoadedShard>,
        index: HashMap<String, TensorIndex>,
        shapes: HashMap<String, Shape>,
    ) -> Self {
        Self {
            artifact_dir,
            shards,
            index,
            shapes,
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

        let dtype = dtype_from_safe(tensor_view.dtype()).ok_or_else(|| Error::Safetensors {
            shard: shard.path.clone(),
            reason: format!("unsupported dtype: {:?}", tensor_view.dtype()),
        })?;

        let shape = self.shapes.get(name).cloned().unwrap_or_else(|| {
            // Fallback: create Shape from empty slice if shape not found
            Shape::from_slice(&[]).unwrap_or_else(|_| {
                // This should never fail for empty slice, but handle it just in case
                Shape::from_slice(&[1]).expect("failed to create fallback shape")
            })
        });

        Ok(TensorView {
            dtype,
            shape,
            data: tensor_view.data(),
        })
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
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let shards_dir = dir.join(SHARDS_DIR);
    fs::create_dir_all(&shards_dir).map_err(|e| Error::io(&shards_dir, e))?;

    let plan = plan_shards(tensors, opts.shard_max_bytes)?;

    let mut shard_checksums = Vec::new();
    for shard in &plan.shards {
        let shard_path = dir.join(shard.relative_path());
        write_shard(shard, tensors, &shard_path, opts.alignment_bytes)?;

        if opts.checksum {
            let checksum = compute_file_checksum(&shard_path)?;
            shard_checksums.push(checksum);
        }
    }

    let mut manifest = TensorSetManifest::new();
    manifest.shards = ShardInfo {
        files: plan.shards.iter().map(|s| s.relative_path()).collect(),
        checksums: shard_checksums,
    };

    let tensor_map: HashMap<&str, &TensorToSave<'a>> =
        tensors.iter().map(|t| (t.meta.name.as_str(), t)).collect();

    for (shard_idx, shard) in plan.shards.iter().enumerate() {
        for name in &shard.tensor_names {
            let tensor = tensor_map[name.as_str()];
            let tensor_checksum = if opts.checksum {
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

    let manifest_path = dir.join(TENSOR_SET_MANIFEST_NAME);
    let file = File::create(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &manifest).map_err(|e| Error::ManifestParse {
        path: manifest_path.clone(),
        reason: e.to_string(),
    })?;

    if let Ok(file) = File::open(&manifest_path) {
        let _ = file.sync_all();
    }

    Ok(())
}

pub fn load_tensor_set(dir: &Path, opts: &TensorSetLoadOptions) -> Result<TensorSet> {
    if !dir.exists() {
        return Err(Error::DirectoryNotFound {
            dir: dir.to_path_buf(),
        });
    }

    let manifest_path = dir.join(TENSOR_SET_MANIFEST_NAME);
    if !manifest_path.exists() {
        return Err(Error::ManifestNotFound {
            path: manifest_path,
        });
    }

    let manifest_content =
        fs::read_to_string(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let manifest: TensorSetManifest =
        serde_json::from_str(&manifest_content).map_err(|e| Error::ManifestParse {
            path: manifest_path.clone(),
            reason: e.to_string(),
        })?;

    if manifest.schema_version != TENSOR_SET_SCHEMA_VERSION {
        return Err(Error::SchemaVersionMismatch {
            file: manifest_path,
            expected: TENSOR_SET_SCHEMA_VERSION.to_string(),
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

        let checksum_ok = if i < manifest.shards.checksums.len() {
            let expected = &manifest.shards.checksums[i];
            match verify_checksum(&shard_path, expected) {
                Ok(()) => true,
                Err(e) => {
                    if matches!(opts.error_mode, ErrorMode::Strict) {
                        return Err(e);
                    }
                    corrupted_shards.push(i);
                    false
                }
            }
        } else {
            true
        };

        let shard = if opts.lazy {
            LoadedShard::load_mmap(&shard_path)?
        } else {
            LoadedShard::load_eager(&shard_path)?
        };

        shards.push(shard);
        let _ = checksum_ok;
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
        let shard_idx =
            *shard_path_to_idx
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

    Ok(TensorSet::new(dir.to_path_buf(), shards, index, shapes))
}

pub fn inspect_tensor_set(dir: &Path) -> Result<Vec<TensorMeta>> {
    if !dir.exists() {
        return Err(Error::DirectoryNotFound {
            dir: dir.to_path_buf(),
        });
    }

    let manifest_path = dir.join(TENSOR_SET_MANIFEST_NAME);
    if !manifest_path.exists() {
        return Err(Error::ManifestNotFound {
            path: manifest_path,
        });
    }

    let content = fs::read_to_string(&manifest_path).map_err(|e| Error::io(&manifest_path, e))?;
    let manifest: TensorSetManifest =
        serde_json::from_str(&content).map_err(|e| Error::ManifestParse {
            path: manifest_path.clone(),
            reason: e.to_string(),
        })?;

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


