use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use bolt_core::dtype::DType;
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors, TensorView};

use crate::Result;
use crate::manifest::CheckpointManifest;
use crate::options::{CheckpointOptions, LoadOpts};
use crate::record::{Record, RecordMeta, RecordView};

pub struct SafeTensorsFormatWriter {
    dir: std::path::PathBuf,
    current_shard_id: usize,
    current_shard_records: Vec<Record>,
    shards_dir: std::path::PathBuf,
    all_metadata: Vec<RecordMeta>,
}

impl SafeTensorsFormatWriter {
    pub fn new(dir: &Path, _opts: &CheckpointOptions) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let shards_dir = dir.join("shards");
        std::fs::create_dir_all(&shards_dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            current_shard_id: 0,
            current_shard_records: Vec::new(),
            shards_dir,
            all_metadata: Vec::new(),
        })
    }

    fn dtype_to_safe(dtype: DType) -> SafeDtype {
        match dtype {
            DType::U8 => SafeDtype::U8,
            DType::I32 => SafeDtype::I32,
            DType::I64 => SafeDtype::I64,
            DType::F32 => SafeDtype::F32,
            DType::F64 => SafeDtype::F64,
        }
    }

    fn write_shard(&mut self) -> Result<Vec<RecordMeta>> {
        if self.current_shard_records.is_empty() {
            return Ok(Vec::new());
        }

        // Build tensor map for safetensors
        let mut tensors = HashMap::new();
        for record in &self.current_shard_records {
            let shape = record.shape.as_slice().to_vec();
            let dtype = Self::dtype_to_safe(record.dtype);
            let data = record.data.as_slice();
            let view = TensorView::new(dtype, shape.clone(), data).map_err(|e| {
                crate::Error::Format(format!("Failed to create tensor view: {}", e))
            })?;
            tensors.insert(record.key.clone(), view);
        }

        // Serialize to safetensors format
        let serialized = safetensors::serialize(&tensors, None)
            .map_err(|e| crate::Error::Format(format!("Failed to serialize safetensors: {}", e)))?;

        // Write shard file
        let shard_filename = format!("shard-{:05}.safetensors", self.current_shard_id);
        let shard_path = self.shards_dir.join(&shard_filename);
        let mut file = File::create(&shard_path).map_err(|e| {
            crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create shard file {:?}: {}", shard_path, e),
            ))
        })?;
        file.write_all(&serialized).map_err(|e| {
            crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write shard file {:?}: {}", shard_path, e),
            ))
        })?;
        file.flush()?;

        // Parse safetensors header to extract metadata (offsets)
        // Safetensors format: [8 bytes: header_len][header_json][padding][data...]
        if serialized.len() < 8 {
            return Err(crate::Error::Format(
                "Invalid safetensors file: too short".to_string(),
            ));
        }

        let header_len = u64::from_le_bytes([
            serialized[0],
            serialized[1],
            serialized[2],
            serialized[3],
            serialized[4],
            serialized[5],
            serialized[6],
            serialized[7],
        ]) as usize;

        if serialized.len() < 8 + header_len {
            return Err(crate::Error::Format(
                "Invalid safetensors file: header too short".to_string(),
            ));
        }

        let header_json = std::str::from_utf8(&serialized[8..8 + header_len])
            .map_err(|e| crate::Error::Format(format!("Invalid header JSON: {}", e)))?;

        let header: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| crate::Error::Format(format!("Failed to parse header JSON: {}", e)))?;

        // Calculate data start offset (after header + padding)
        let padding = (8 - (header_len % 8)) % 8;
        let data_start = 8 + header_len + padding;

        let mut metadata = Vec::new();
        for record in &self.current_shard_records {
            let tensor_info = header
                .get(&record.key)
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    crate::Error::Format(format!("Tensor {} not found in header", record.key))
                })?;

            let data_offsets = tensor_info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    crate::Error::Format(format!("Invalid data_offsets for {}", record.key))
                })?;

            if data_offsets.len() != 2 {
                return Err(crate::Error::Format(format!(
                    "Invalid data_offsets length for {}",
                    record.key
                )));
            }

            let offset = data_offsets[0].as_u64().ok_or_else(|| {
                crate::Error::Format(format!("Invalid offset for {}", record.key))
            })?;
            let end = data_offsets[1]
                .as_u64()
                .ok_or_else(|| crate::Error::Format(format!("Invalid end for {}", record.key)))?;

            let meta = RecordMeta {
                key: record.key.clone(),
                dtype: record.dtype,
                shape: record.shape.clone(),
                offset: data_start as u64 + offset,
                length: end - offset,
                shard_id: self.current_shard_id,
            };
            metadata.push(meta);
        }

        // Clear records for next shard
        self.current_shard_records.clear();
        self.current_shard_id += 1;
        Ok(metadata)
    }
}

impl crate::format::FormatWriter for SafeTensorsFormatWriter {
    fn write_record(&mut self, record: Record) -> Result<()> {
        self.current_shard_records.push(record);
        Ok(())
    }

    fn flush_shard(&mut self) -> Result<()> {
        let metadata = self.write_shard()?;
        self.all_metadata.extend(metadata);
        Ok(())
    }

    fn finish(mut self: Box<Self>, manifest: &mut CheckpointManifest) -> Result<()> {
        // Flush any remaining records
        self.flush_shard()?;

        // Add all metadata to manifest
        for meta in self.all_metadata {
            manifest.add_record(meta);
        }

        manifest.shard_count = self.current_shard_id;
        Ok(())
    }
}

pub struct SafeTensorsFormatReader {
    _dir: std::path::PathBuf,
    manifest: CheckpointManifest,
    shards: std::collections::HashMap<usize, (std::path::PathBuf, memmap2::Mmap)>,
}

impl SafeTensorsFormatReader {
    pub fn new(dir: &Path, manifest: &CheckpointManifest, _opts: &LoadOpts) -> Result<Self> {
        let shards_dir = dir.join("shards");
        let mut shards = std::collections::HashMap::new();

        // Pre-load all shard files with memory mapping
        for shard_id in 0..manifest.shard_count {
            let shard_filename = format!("shard-{:05}.safetensors", shard_id);
            let shard_path = shards_dir.join(&shard_filename);

            if shard_path.exists() {
                let file = std::fs::File::open(&shard_path)?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                shards.insert(shard_id, (shard_path, mmap));
            }
        }

        Ok(Self {
            _dir: dir.to_path_buf(),
            manifest: manifest.clone(),
            shards,
        })
    }

    fn safe_dtype_to_dtype(safe_dtype: SafeDtype) -> Option<DType> {
        match safe_dtype {
            SafeDtype::U8 => Some(DType::U8),
            SafeDtype::I32 => Some(DType::I32),
            SafeDtype::I64 => Some(DType::I64),
            SafeDtype::F32 => Some(DType::F32),
            SafeDtype::F64 => Some(DType::F64),
            _ => None,
        }
    }
}

impl crate::format::FormatReader for SafeTensorsFormatReader {
    fn read_record(&self, key: &str) -> Result<RecordView> {
        let meta = self
            .manifest
            .get_record(key)
            .ok_or_else(|| crate::Error::KeyNotFound(key.to_string()))?;

        // Get the shard for this record
        let (shard_path, mmap) = self.shards.get(&meta.shard_id).ok_or_else(|| {
            crate::Error::InvalidCheckpoint(format!("Shard {} not found", meta.shard_id))
        })?;

        // Deserialize safetensors from memory-mapped file
        let safe_tensors = SafeTensors::deserialize(mmap.as_ref()).map_err(|e| {
            crate::Error::Format(format!(
                "Failed to deserialize safetensors from {:?}: {}",
                shard_path, e
            ))
        })?;

        // Get tensor view to validate it exists
        safe_tensors
            .tensor(key)
            .map_err(|_| crate::Error::KeyNotFound(key.to_string()))?;

        // Extract data slice using metadata offsets (lazy - just reference the mmap)
        let data_start = meta.offset as usize;
        let data_end = (meta.offset + meta.length) as usize;

        if data_end > mmap.len() {
            return Err(crate::Error::InvalidCheckpoint(format!(
                "Data range [{}, {}) exceeds shard size {}",
                data_start,
                data_end,
                mmap.len()
            )));
        }

        let data = mmap[data_start..data_end].to_vec();

        Ok(RecordView {
            meta: meta.clone(),
            data,
        })
    }

    fn contains(&self, key: &str) -> bool {
        self.manifest.get_record(key).is_some()
    }

    fn list_keys(&self) -> Vec<String> {
        self.manifest.records.keys().cloned().collect()
    }
}
