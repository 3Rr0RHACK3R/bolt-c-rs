use std::collections::BTreeMap;

use bolt_core::DType;
use bolt_core::shape::Shape;
use serde::{Deserialize, Serialize};

use crate::TensorRole;
use crate::utils::now_rfc3339;

pub const TENSOR_SET_SCHEMA_VERSION: &str = "bolt-tensorset:1";
pub const CHECKPOINT_SCHEMA_VERSION: &str = "bolt-checkpoint:1";
pub const MIN_READER_VERSION: &str = "0.1.0";

pub const TENSOR_SET_MANIFEST_NAME: &str = "bolt-tensorset.json";
pub const CHECKPOINT_MANIFEST_NAME: &str = "bolt-checkpoint.json";
pub const SHARDS_DIR: &str = "shards";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorLocation {
    pub shard: String,
    pub key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorEntry {
    pub role: TensorRole,
    pub group: u32,
    pub dtype: String,
    #[serde(with = "crate::serde_shape")]
    pub shape: Shape,
    pub location: TensorLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

impl TensorEntry {
    pub fn parse_dtype(&self) -> Option<DType> {
        DType::from_name(&self.dtype)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub checksums: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorSetManifest {
    pub schema_version: String,
    pub created_at: String,
    pub min_reader_version: String,
    pub shards: ShardInfo,
    pub tensors: BTreeMap<String, TensorEntry>,
}

impl TensorSetManifest {
    pub fn new() -> Self {
        Self {
            schema_version: TENSOR_SET_SCHEMA_VERSION.to_string(),
            created_at: now_rfc3339(),
            min_reader_version: MIN_READER_VERSION.to_string(),
            shards: ShardInfo {
                files: Vec::new(),
                checksums: Vec::new(),
            },
            tensors: BTreeMap::new(),
        }
    }
}

impl Default for TensorSetManifest {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CheckpointMetadataJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_step: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default)]
    pub user: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub schema_version: String,
    pub created_at: String,
    pub min_reader_version: String,
    pub metadata: CheckpointMetadataJson,
    pub shards: ShardInfo,
    pub tensors: BTreeMap<String, TensorEntry>,
}

impl CheckpointManifest {
    pub fn new() -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_string(),
            created_at: now_rfc3339(),
            min_reader_version: MIN_READER_VERSION.to_string(),
            metadata: CheckpointMetadataJson::default(),
            shards: ShardInfo {
                files: Vec::new(),
                checksums: Vec::new(),
            },
            tensors: BTreeMap::new(),
        }
    }
}

impl Default for CheckpointManifest {
    fn default() -> Self {
        Self::new()
    }
}

pub fn dtype_to_name(dtype: DType) -> &'static str {
    dtype.name()
}
