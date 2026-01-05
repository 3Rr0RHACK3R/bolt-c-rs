//! Shard loading and storage - handles reading safetensors files from disk.

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype as SafeDtype;

use bolt_core::DType;

use crate::{Error, OnError, Result};

pub fn compute_file_checksum(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path).map_err(|e| Error::io(path, e))?);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 65536];

    loop {
        let n = file.read(&mut buffer).map_err(|e| Error::io(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let hash = hasher.finalize();
    Ok(format!("b3:{}", hash.to_hex()))
}

pub fn verify_checksum(path: &Path, expected: &str) -> Result<()> {
    let computed = compute_file_checksum(path)?;
    if computed != expected {
        return Err(Error::ShardChecksumMismatch {
            shard: path.to_path_buf(),
            expected: expected.to_string(),
            computed,
        });
    }
    Ok(())
}

pub enum ShardStorage {
    Eager(Vec<u8>),
    Mapped(Mmap),
}

impl ShardStorage {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ShardStorage::Eager(v) => v,
            ShardStorage::Mapped(m) => m,
        }
    }
}

pub struct LoadedShard {
    pub path: PathBuf,
    pub storage: ShardStorage,
}

impl LoadedShard {
    pub fn load_eager(path: &Path) -> Result<Self> {
        let data = fs::read(path).map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            storage: ShardStorage::Eager(data),
        })
    }

    pub fn load_mmap(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            storage: ShardStorage::Mapped(mmap),
        })
    }

    pub fn tensors(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(self.storage.as_bytes()).map_err(|e| Error::ShardFormat {
            shard: self.path.clone(),
            reason: e.to_string(),
        })
    }
}

pub fn dtype_from_safe(dtype: SafeDtype) -> Option<DType> {
    match dtype {
        SafeDtype::U8 => Some(DType::U8),
        SafeDtype::I32 => Some(DType::I32),
        SafeDtype::I64 => Some(DType::I64),
        SafeDtype::F32 => Some(DType::F32),
        SafeDtype::F64 => Some(DType::F64),
        _ => None,
    }
}

pub fn load_shards_with_verification(
    shard_files: &[String],
    shard_checksums: &[String],
    dir: &Path,
    lazy: bool,
    on_error: OnError,
) -> Result<(Vec<LoadedShard>, Vec<usize>)> {
    let mut shards = Vec::new();
    let mut corrupted_shard_indices = Vec::new();

    for (shard_index, shard_rel_path) in shard_files.iter().enumerate() {
        validate_shard_path(shard_rel_path, dir)?;
        let shard_path = dir.join(shard_rel_path);

        if !shard_path.exists() {
            return Err(Error::ShardNotFound { path: shard_path });
        }

        if let Some(expected) = shard_checksums.get(shard_index) {
            match verify_checksum(&shard_path, expected) {
                Ok(()) => {}
                Err(e) => {
                    if matches!(on_error, OnError::Fail) {
                        return Err(e);
                    }
                    corrupted_shard_indices.push(shard_index);
                }
            }
        }

        let shard = if lazy {
            LoadedShard::load_mmap(&shard_path)?
        } else {
            LoadedShard::load_eager(&shard_path)?
        };

        shards.push(shard);
    }

    Ok((shards, corrupted_shard_indices))
}

fn validate_shard_path(path: &str, base_dir: &Path) -> Result<()> {
    if path.is_empty() {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot be empty".to_string(),
        });
    }

    if path.starts_with('/') || path.starts_with('\\') {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot be absolute".to_string(),
        });
    }

    if path.contains("..") {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot contain '..' (parent directory escape)".to_string(),
        });
    }

    let full_path = base_dir.join(path);
    let canonical_base = base_dir
        .canonicalize()
        .ok()
        .unwrap_or_else(|| base_dir.to_path_buf());
    let canonical_full = full_path.canonicalize().ok();

    if let Some(ref canonical) = canonical_full
        && !canonical.starts_with(&canonical_base)
    {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: format!(
                "resolved path escapes artifact directory {:?}",
                canonical_base
            ),
        });
    }

    Ok(())
}
