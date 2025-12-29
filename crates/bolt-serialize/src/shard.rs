use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Dtype as SafeDtype, TensorView as SafeTensorView};
use safetensors::{SafeTensors, serialize};

use bolt_core::DType;

use crate::manifest::SHARDS_DIR;
use crate::{Error, Result, TensorToSave};

const DEFAULT_SHARD_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB

#[derive(Clone, Debug)]
pub struct ShardPlan {
    pub shards: Vec<PlannedShard>,
}

#[derive(Clone, Debug)]
pub struct PlannedShard {
    pub index: usize,
    pub total: usize,
    pub tensor_names: Vec<String>,
}

impl PlannedShard {
    pub fn filename(&self) -> String {
        format!(
            "weights-{:05}-of-{:05}.safetensors",
            self.index + 1,
            self.total
        )
    }

    pub fn relative_path(&self) -> String {
        format!("{}/{}", SHARDS_DIR, self.filename())
    }
}

pub fn plan_shards<'a>(tensors: &[TensorToSave<'a>], max_bytes: Option<u64>) -> Result<ShardPlan> {
    let max_bytes = max_bytes.unwrap_or(DEFAULT_SHARD_MAX_BYTES);

    let mut tensor_indices_by_name: Vec<usize> = (0..tensors.len()).collect();
    tensor_indices_by_name.sort_by(|&a, &b| tensors[a].meta.name.cmp(&tensors[b].meta.name));

    let mut shards: Vec<PlannedShard> = Vec::new();
    let mut shard_tensor_names: Vec<String> = Vec::new();
    let mut shard_bytes: u64 = 0;

    for &tensor_index in &tensor_indices_by_name {
        let tensor = &tensors[tensor_index];
        let tensor_bytes = tensor.data.len() as u64;

        if tensor_bytes > max_bytes {
            if !shard_tensor_names.is_empty() {
                shards.push(PlannedShard {
                    index: shards.len(),
                    total: 0, // will be updated later
                    tensor_names: std::mem::take(&mut shard_tensor_names),
                });
                shard_bytes = 0;
            }
            shards.push(PlannedShard {
                index: shards.len(),
                total: 0,
                tensor_names: vec![tensor.meta.name.clone()],
            });
            continue;
        }

        if shard_bytes + tensor_bytes > max_bytes && !shard_tensor_names.is_empty() {
            shards.push(PlannedShard {
                index: shards.len(),
                total: 0,
                tensor_names: std::mem::take(&mut shard_tensor_names),
            });
            shard_bytes = 0;
        }

        shard_tensor_names.push(tensor.meta.name.clone());
        shard_bytes += tensor_bytes;
    }

    if !shard_tensor_names.is_empty() {
        shards.push(PlannedShard {
            index: shards.len(),
            total: 0,
            tensor_names: shard_tensor_names,
        });
    }

    let total = shards.len();
    for shard in &mut shards {
        shard.total = total;
    }

    Ok(ShardPlan { shards })
}

pub fn write_shard<'a>(
    shard: &PlannedShard,
    tensors: &[TensorToSave<'a>],
    out_path: &Path,
    alignment_bytes: u64,
) -> Result<()> {
    let tensor_map: HashMap<&str, &TensorToSave<'a>> =
        tensors.iter().map(|t| (t.meta.name.as_str(), t)).collect();

    // Build safetensors data
    let mut views: Vec<(&str, SafeTensorView<'_>)> = Vec::with_capacity(shard.tensor_names.len());

    for name in &shard.tensor_names {
        let tensor = tensor_map
            .get(name.as_str())
            .ok_or_else(|| Error::TensorNotFound {
                name: name.clone(),
                dir: out_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            })?;

        let safe_dtype = dtype_to_safe(tensor.meta.dtype);
        let shape: Vec<usize> = tensor.meta.shape.as_slice().to_vec();

        let view = SafeTensorView::new(safe_dtype, shape, &tensor.data).map_err(|e| {
            Error::Safetensors {
                shard: out_path.to_path_buf(),
                reason: e.to_string(),
            }
        })?;

        views.push((name.as_str(), view));
    }

    // Serialize to bytes (safetensors already pads header to 8-byte alignment)
    let bytes = serialize(views, None).map_err(|e| Error::Safetensors {
        shard: out_path.to_path_buf(),
        reason: e.to_string(),
    })?;

    // Apply additional alignment padding if requested
    // The safetensors format is: 8-byte header size + N-byte JSON header + tensor data
    // Safetensors pads the JSON header to 8-byte alignment.
    // For page-aligned access (e.g., 4096 bytes), we pad the entire file start.
    let aligned_bytes = if alignment_bytes > 8 {
        pad_for_alignment(&bytes, alignment_bytes)
    } else {
        bytes
    };

    // Write to file
    let mut file = BufWriter::new(File::create(out_path).map_err(|e| Error::io(out_path, e))?);
    file.write_all(&aligned_bytes)
        .map_err(|e| Error::io(out_path, e))?;
    file.flush().map_err(|e| Error::io(out_path, e))?;

    Ok(())
}

fn pad_for_alignment(data: &[u8], alignment: u64) -> Vec<u8> {
    if data.len() < 8 {
        return data.to_vec();
    }

    let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let current_data_offset = 8 + header_size;

    // Calculate padding needed
    let alignment = alignment.max(1);
    let remainder = current_data_offset % alignment;
    if remainder == 0 {
        return data.to_vec();
    }

    let padding_needed = (alignment - remainder) as usize;

    let new_header_size = header_size + padding_needed as u64;
    let mut result = Vec::with_capacity(data.len() + padding_needed);

    result.extend_from_slice(&new_header_size.to_le_bytes());

    let original_header = &data[8..(8 + header_size as usize)];
    result.extend_from_slice(original_header);

    result.extend(std::iter::repeat(b' ').take(padding_needed));

    result.extend_from_slice(&data[(8 + header_size as usize)..]);

    result
}

pub fn compute_file_checksum(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path).map_err(|e| Error::io(path, e))?);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 65536]; // 64KB buffer

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

pub fn compute_bytes_checksum(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    format!("b3:{}", hash.to_hex())
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
        SafeTensors::deserialize(self.storage.as_bytes()).map_err(|e| Error::Safetensors {
            shard: self.path.clone(),
            reason: e.to_string(),
        })
    }
}

pub fn dtype_to_safe(dtype: DType) -> SafeDtype {
    match dtype {
        DType::U8 => SafeDtype::U8,
        DType::I32 => SafeDtype::I32,
        DType::I64 => SafeDtype::I64,
        DType::F32 => SafeDtype::F32,
        DType::F64 => SafeDtype::F64,
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
