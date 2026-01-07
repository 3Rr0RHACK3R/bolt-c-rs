use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use bolt_core::BaseBackend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::DType;
use bolt_core::dtype::Float;
use bolt_core::shape::Shape;
use bolt_tensor::Tensor;
use serde::Serialize;

use crate::Result;
use crate::format::FormatKind;
use crate::format::FormatWriter;
use crate::manifest::CheckpointManifest;
use crate::manifest::CheckpointMeta;
use crate::options::CheckpointOptions;
use crate::record::Record;

pub struct CheckpointWriter {
    dir: std::path::PathBuf,
    format: Box<dyn FormatWriter>,
    format_kind: FormatKind,
    prefix_stack: Vec<String>,
    current_shard: Vec<Record>,
    current_shard_bytes: usize,
    shard_max_bytes: usize,
    seen_keys: HashSet<String>,
}

impl CheckpointWriter {
    pub fn new(dir: &Path, opts: &CheckpointOptions) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let format = opts.format.create_writer(dir, opts)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            format,
            format_kind: opts.format.clone(),
            prefix_stack: Vec::new(),
            current_shard: Vec::new(),
            current_shard_bytes: 0,
            shard_max_bytes: opts.shard_max_bytes,
            seen_keys: HashSet::new(),
        })
    }

    fn make_key(&self, key: &str) -> String {
        if self.prefix_stack.is_empty() {
            key.to_string()
        } else {
            format!("{}.{}", self.prefix_stack.join("."), key)
        }
    }

    fn check_duplicate(&mut self, key: &str) -> Result<()> {
        if !self.seen_keys.insert(key.to_string()) {
            return Err(crate::Error::DuplicateKey(key.to_string()));
        }
        Ok(())
    }

    fn add_record(&mut self, record: Record) -> Result<()> {
        let record_bytes = record.size_bytes();

        // Flush shard if adding this record would exceed the limit
        if self.current_shard_bytes > 0
            && self.current_shard_bytes + record_bytes > self.shard_max_bytes
        {
            self.flush_current_shard()?;
        }

        self.current_shard_bytes += record_bytes;
        self.current_shard.push(record);
        Ok(())
    }

    fn flush_current_shard(&mut self) -> Result<()> {
        if self.current_shard.is_empty() {
            return Ok(());
        }

        // Write all records in current shard to format
        for record in self.current_shard.drain(..) {
            self.format.write_record(record)?;
        }

        // Flush the shard in the format
        self.format.flush_shard()?;
        self.current_shard_bytes = 0;
        Ok(())
    }

    pub fn with_prefix<F>(&mut self, prefix: &str, f: F) -> Result<()>
    where
        F: FnOnce(&mut Self) -> Result<()>,
    {
        self.prefix_stack.push(prefix.to_string());
        let result = f(self);
        self.prefix_stack.pop();
        result
    }

    pub fn save_prefixed(&mut self, prefix: &str, item: &dyn crate::SaveCheckpoint) -> Result<()> {
        self.with_prefix(prefix, |w| item.save(w))
    }

    pub fn tensor<B, D>(&mut self, key: &str, t: &Tensor<B, D>) -> Result<()>
    where
        B: BaseBackend + CopyOp<D>,
        D: Float + bytemuck::Pod,
    {
        let full_key = self.make_key(key);
        self.check_duplicate(&full_key)?;

        // Read tensor data
        let data_vec = t
            .to_vec()
            .map_err(|e| crate::Error::Serialization(format!("Failed to read tensor: {}", e)))?;
        let shape = t.shape().clone();
        let dtype = D::DTYPE;

        // Convert to bytes
        let data = bytemuck::cast_slice(&data_vec).to_vec();

        let record = Record::new(full_key, dtype, shape, data);
        self.add_record(record)
    }

    pub fn bytes(&mut self, key: &str, data: Vec<u8>, dtype: DType, shape: &Shape) -> Result<()> {
        let full_key = self.make_key(key);
        self.check_duplicate(&full_key)?;

        let record = Record::new(full_key, dtype, shape.clone(), data);
        self.add_record(record)
    }

    pub fn u64(&mut self, key: &str, v: u64) -> Result<()> {
        self.json(key, &v)
    }

    pub fn f32(&mut self, key: &str, v: f32) -> Result<()> {
        let full_key = self.make_key(key);
        self.check_duplicate(&full_key)?;

        let data = v.to_le_bytes().to_vec();
        let shape = Shape::from_slice(&[])
            .map_err(|e| crate::Error::Serialization(format!("Invalid shape: {}", e)))?;
        let record = Record::new(full_key, DType::F32, shape, data);
        self.add_record(record)
    }

    pub fn i64(&mut self, key: &str, v: i64) -> Result<()> {
        let full_key = self.make_key(key);
        self.check_duplicate(&full_key)?;

        let data = v.to_le_bytes().to_vec();
        let shape = Shape::from_slice(&[])
            .map_err(|e| crate::Error::Serialization(format!("Invalid shape: {}", e)))?;
        let record = Record::new(full_key, DType::I64, shape, data);
        self.add_record(record)
    }

    pub fn json<T: Serialize>(&mut self, key: &str, v: &T) -> Result<()> {
        let full_key = self.make_key(key);
        self.check_duplicate(&full_key)?;

        let json_bytes = serde_json::to_vec(v).map_err(|e| {
            crate::Error::Serialization(format!("JSON serialization failed: {}", e))
        })?;
        let shape = Shape::from_slice(&[json_bytes.len()])
            .map_err(|e| crate::Error::Serialization(format!("Invalid shape: {}", e)))?;
        let record = Record::new(full_key, DType::U8, shape, json_bytes);
        self.add_record(record)
    }

    pub fn tensor_map<B, D, K, F>(
        &mut self,
        prefix: &str,
        map: &HashMap<K, Tensor<B, D>>,
        key_fn: F,
    ) -> Result<()>
    where
        B: BaseBackend + CopyOp<D>,
        D: Float + bytemuck::Pod,
        F: Fn(&K) -> String,
    {
        self.with_prefix(prefix, |w| {
            for (k, tensor) in map {
                let key = key_fn(k);
                w.tensor(&key, tensor)?;
            }
            Ok(())
        })
    }

    pub fn finish(mut self, meta: &CheckpointMeta) -> Result<()> {
        // Flush any remaining records
        self.flush_current_shard()?;

        // Create manifest
        let mut manifest = CheckpointManifest::new(meta.clone(), self.format_kind.clone());

        // Finish format writing (this will populate manifest with record metadata)
        self.format.finish(&mut manifest)?;

        // Write manifest to disk
        let manifest_path = self.dir.join("bolt-checkpoint.json");
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize manifest: {}", e))
        })?;
        std::fs::write(&manifest_path, manifest_json)?;

        Ok(())
    }
}
