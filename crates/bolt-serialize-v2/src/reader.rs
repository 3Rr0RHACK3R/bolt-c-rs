use std::path::Path;
use std::sync::Arc;

use bolt_core::BaseBackend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::DType;
use bolt_core::dtype::Float;
use bolt_tensor::Tensor;
use bytemuck::cast_slice;
use serde::de::DeserializeOwned;

use crate::Result;
use crate::format::FormatKind;
use crate::format::FormatReader;
use crate::manifest::CheckpointInfo;
use crate::manifest::CheckpointManifest;
use crate::options::LoadOpts;
use crate::record::RecordMeta;

pub struct CheckpointReader {
    format: Box<dyn FormatReader>,
    manifest: CheckpointManifest,
    info: CheckpointInfo,
    prefix_stack: Vec<String>,
}

impl CheckpointReader {
    pub fn open(dir: &Path, opts: &LoadOpts) -> Result<Self> {
        // Read manifest
        let manifest_path = dir.join("bolt-checkpoint.json");
        let manifest_json = std::fs::read_to_string(&manifest_path).map_err(|e| {
            crate::Error::InvalidCheckpoint(format!("Failed to read manifest: {}", e))
        })?;
        let manifest: CheckpointManifest = serde_json::from_str(&manifest_json).map_err(|e| {
            crate::Error::InvalidCheckpoint(format!("Failed to parse manifest: {}", e))
        })?;

        // Detect format from manifest
        let format_kind = FormatKind::from_manifest(&manifest);

        // Create format reader
        let format = format_kind.create_reader(dir, &manifest, opts)?;

        let info = CheckpointInfo::from(&manifest);

        Ok(Self {
            format,
            manifest,
            info,
            prefix_stack: Vec::new(),
        })
    }

    fn make_key(&self, key: &str) -> String {
        if self.prefix_stack.is_empty() {
            key.to_string()
        } else {
            format!("{}.{}", self.prefix_stack.join("."), key)
        }
    }

    pub fn with_prefix<F, R>(&mut self, prefix: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        self.prefix_stack.push(prefix.to_string());
        let result = f(self);
        self.prefix_stack.pop();
        result
    }

    pub fn load_prefixed(
        &mut self,
        prefix: &str,
        item: &mut dyn crate::LoadCheckpoint,
    ) -> Result<()> {
        self.with_prefix(prefix, |r| item.load(r))
    }

    pub fn tensor<B, D>(&self, key: &str, backend: &Arc<B>) -> Result<Tensor<B, D>>
    where
        B: BaseBackend + CopyOp<D>,
        D: Float + bytemuck::Pod,
    {
        let full_key = self.make_key(key);
        let view = self.format.read_record(&full_key)?;

        // Validate dtype
        if view.meta.dtype != D::DTYPE {
            return Err(crate::Error::InvalidDtype {
                expected: format!("{:?}", D::DTYPE),
                got: format!("{:?}", view.meta.dtype),
            });
        }

        // Validate shape
        let expected_shape = view.meta.shape.clone();
        let numel = expected_shape.num_elements();
        let expected_bytes = numel * D::DTYPE.size_in_bytes();

        if view.data.len() != expected_bytes {
            return Err(crate::Error::InvalidCheckpoint(format!(
                "Data size mismatch for key {}: expected {} bytes, got {}",
                full_key,
                expected_bytes,
                view.data.len()
            )));
        }

        // Convert bytes to typed data
        let data: Vec<D> = bytemuck::cast_slice(&view.data).to_vec();

        // Create tensor
        Tensor::from_vec(backend, data, expected_shape.as_slice())
            .map_err(|e| crate::Error::Deserialization(format!("Failed to create tensor: {}", e)))
    }

    pub fn bytes(&self, key: &str) -> Result<Vec<u8>> {
        let full_key = self.make_key(key);
        let view = self.format.read_record(&full_key)?;
        Ok(view.data)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.json(key)
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        let full_key = self.make_key(key);
        let view = self.format.read_record(&full_key)?;

        if view.meta.dtype != DType::F32 {
            return Err(crate::Error::InvalidDtype {
                expected: "F32".to_string(),
                got: format!("{:?}", view.meta.dtype),
            });
        }

        if view.data.len() != 4 {
            return Err(crate::Error::InvalidCheckpoint(format!(
                "Invalid f32 data size: expected 4 bytes, got {}",
                view.data.len()
            )));
        }

        Ok(f32::from_le_bytes([
            view.data[0],
            view.data[1],
            view.data[2],
            view.data[3],
        ]))
    }

    pub fn i64(&self, key: &str) -> Result<i64> {
        let full_key = self.make_key(key);
        let view = self.format.read_record(&full_key)?;

        if view.meta.dtype != DType::I64 {
            return Err(crate::Error::InvalidDtype {
                expected: "I64".to_string(),
                got: format!("{:?}", view.meta.dtype),
            });
        }

        if view.data.len() != 8 {
            return Err(crate::Error::InvalidCheckpoint(format!(
                "Invalid i64 data size: expected 8 bytes, got {}",
                view.data.len()
            )));
        }

        Ok(i64::from_le_bytes([
            view.data[0],
            view.data[1],
            view.data[2],
            view.data[3],
            view.data[4],
            view.data[5],
            view.data[6],
            view.data[7],
        ]))
    }

    pub fn json<T: DeserializeOwned>(&self, key: &str) -> Result<T> {
        let full_key = self.make_key(key);
        let view = self.format.read_record(&full_key)?;

        serde_json::from_slice(&view.data).map_err(|e| {
            crate::Error::Deserialization(format!("Failed to deserialize JSON: {}", e))
        })
    }

    pub fn contains(&self, key: &str) -> bool {
        let full_key = self.make_key(key);
        self.format.contains(&full_key)
    }

    pub fn keys(&self) -> Vec<String> {
        let all_keys = self.format.list_keys();

        if self.prefix_stack.is_empty() {
            return all_keys;
        }

        let prefix = self.prefix_stack.join(".");
        let prefix_with_dot = if prefix.is_empty() {
            String::new()
        } else {
            format!("{}.", prefix)
        };

        all_keys
            .into_iter()
            .filter(|k| k.starts_with(&prefix_with_dot) || k.as_str() == prefix.as_str())
            .map(|k| {
                if k.as_str() == prefix.as_str() {
                    String::new()
                } else if k.len() > prefix_with_dot.len() {
                    k[prefix_with_dot.len()..].to_string()
                } else {
                    k
                }
            })
            .collect()
    }

    pub fn keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        let full_prefix = self.make_key(prefix);
        self.format
            .list_keys()
            .into_iter()
            .filter(|k| k.starts_with(&full_prefix))
            .collect()
    }

    pub fn metadata(&self, key: &str) -> Option<&RecordMeta> {
        let full_key = self.make_key(key);
        self.manifest.get_record(&full_key)
    }

    pub fn info(&self) -> &CheckpointInfo {
        &self.info
    }
}
