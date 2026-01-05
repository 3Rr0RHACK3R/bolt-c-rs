use std::fs::File;
use std::io::Write;
use std::path::Path;

use bolt_core::dtype::DType;

use crate::Result;
use crate::manifest::CheckpointManifest;
use crate::options::{CheckpointOptions, LoadOpts};
use crate::record::{Record, RecordMeta, RecordView};

const BINARY_MAGIC: u64 = 0x424F4C545F5632; // "BOLT_V2"

pub struct BinaryFormatWriter {
    dir: std::path::PathBuf,
    current_shard_id: usize,
    current_shard_records: Vec<Record>,
    shards_dir: std::path::PathBuf,
    all_metadata: Vec<RecordMeta>,
}

impl BinaryFormatWriter {
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

    fn dtype_to_u8(dtype: DType) -> u8 {
        match dtype {
            DType::U8 => 0,
            DType::I32 => 1,
            DType::I64 => 2,
            DType::F32 => 3,
            DType::F64 => 4,
        }
    }

    fn u8_to_dtype(byte: u8) -> Option<DType> {
        match byte {
            0 => Some(DType::U8),
            1 => Some(DType::I32),
            2 => Some(DType::I64),
            3 => Some(DType::F32),
            4 => Some(DType::F64),
            _ => None,
        }
    }

    fn write_shard(&mut self) -> Result<Vec<RecordMeta>> {
        if self.current_shard_records.is_empty() {
            return Ok(Vec::new());
        }

        let shard_filename = format!("shard-{:05}.bin", self.current_shard_id);
        let shard_path = self.shards_dir.join(&shard_filename);
        let mut file = File::create(&shard_path)?;

        // Write magic number
        file.write_all(&BINARY_MAGIC.to_le_bytes())?;

        // Write record count
        let record_count = self.current_shard_records.len() as u64;
        file.write_all(&record_count.to_le_bytes())?;

        // Calculate data offset (after header and metadata)
        let mut data_offset = 8 + 8; // magic + record_count
        for record in &self.current_shard_records {
            // key_len (u64) + key bytes + dtype (u8) + shape_len (u64) + shape (u64[]) + data_len (u64)
            data_offset += 8 + record.key.len() + 1 + 8 + (record.shape.as_slice().len() * 8) + 8;
        }

        let mut metadata = Vec::new();
        let mut current_data_offset = data_offset as u64;

        // Write metadata for each record
        for record in &self.current_shard_records {
            // Write key
            let key_bytes = record.key.as_bytes();
            file.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
            file.write_all(key_bytes)?;

            // Write dtype
            file.write_all(&[Self::dtype_to_u8(record.dtype)])?;

            // Write shape
            let shape = record.shape.as_slice();
            file.write_all(&(shape.len() as u64).to_le_bytes())?;
            for &dim in shape {
                file.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Write data length
            let data_len = record.data.len() as u64;
            file.write_all(&data_len.to_le_bytes())?;

            // Create metadata
            let meta = RecordMeta {
                key: record.key.clone(),
                dtype: record.dtype,
                shape: record.shape.clone(),
                offset: current_data_offset,
                length: data_len,
                shard_id: self.current_shard_id,
            };
            metadata.push(meta.clone());
            current_data_offset += data_len;
        }

        // Write data blocks
        for record in &self.current_shard_records {
            file.write_all(&record.data)?;
        }

        file.flush()?;
        self.current_shard_records.clear();
        self.current_shard_id += 1;
        Ok(metadata)
    }
}

impl crate::format::FormatWriter for BinaryFormatWriter {
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

pub struct BinaryFormatReader {
    _dir: std::path::PathBuf,
    manifest: CheckpointManifest,
    shards: std::collections::HashMap<usize, (std::path::PathBuf, memmap2::Mmap)>,
}

impl BinaryFormatReader {
    pub fn new(dir: &Path, manifest: &CheckpointManifest, _opts: &LoadOpts) -> Result<Self> {
        let shards_dir = dir.join("shards");
        let mut shards = std::collections::HashMap::new();

        // Pre-load all shard files with memory mapping
        for shard_id in 0..manifest.shard_count {
            let shard_filename = format!("shard-{:05}.bin", shard_id);
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

    fn u8_to_dtype(byte: u8) -> Option<DType> {
        match byte {
            0 => Some(DType::U8),
            1 => Some(DType::I32),
            2 => Some(DType::I64),
            3 => Some(DType::F32),
            4 => Some(DType::F64),
            _ => None,
        }
    }
}

impl crate::format::FormatReader for BinaryFormatReader {
    fn read_record(&self, key: &str) -> Result<RecordView> {
        let meta = self
            .manifest
            .get_record(key)
            .ok_or_else(|| crate::Error::KeyNotFound(key.to_string()))?;

        // Get the shard for this record
        let (_shard_path, mmap) = self.shards.get(&meta.shard_id).ok_or_else(|| {
            crate::Error::InvalidCheckpoint(format!("Shard {} not found", meta.shard_id))
        })?;

        // Extract data slice (lazy - just reference the mmap)
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
