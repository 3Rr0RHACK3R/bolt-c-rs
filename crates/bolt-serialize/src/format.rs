//! Checkpoint format - manifest schema, serde helpers, and directory structure.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use bolt_core::shape::Shape;
use bolt_core::DType;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, Result, Role};

pub const CHECKPOINT_SCHEMA_VERSION: &str = "bolt-checkpoint:1";
pub const MIN_READER_VERSION: &str = "0.1.0";
pub const CHECKPOINT_MANIFEST_NAME: &str = "bolt-checkpoint.json";
pub const SHARDS_DIR: &str = "shards";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordLocation {
    pub shard: String,
    pub key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordEntry {
    pub role: Role,
    pub group: u32,
    pub dtype: String,
    #[serde(serialize_with = "serialize_shape", deserialize_with = "deserialize_shape")]
    pub shape: Shape,
    pub location: RecordLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param_id: Option<u64>,
}

impl RecordEntry {
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
    pub written_at: String,
    pub min_reader_version: String,
    pub metadata: CheckpointMetadataJson,
    pub shards: ShardInfo,
    pub tensors: BTreeMap<String, RecordEntry>,
}

impl CheckpointManifest {
    pub fn new() -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_string(),
            written_at: now_rfc3339(),
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

pub fn write_manifest(manifest: &impl Serialize, path: &Path) -> Result<()> {
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, manifest).map_err(|e| Error::ManifestParse {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;

    Ok(())
}

pub fn ensure_shards_dir(dir: &Path) -> Result<()> {
    let shards_dir = dir.join(SHARDS_DIR);
    fs::create_dir_all(&shards_dir).map_err(|e| Error::io(&shards_dir, e))?;
    Ok(())
}

fn serialize_shape<S>(shape: &Shape, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    shape.as_slice().serialize(serializer)
}

fn deserialize_shape<'de, D>(deserializer: D) -> std::result::Result<Shape, D::Error>
where
    D: Deserializer<'de>,
{
    let dims: Vec<usize> = Vec::deserialize(deserializer)?;
    Shape::from_slice(&dims).map_err(de::Error::custom)
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .expect("RFC3339 formatting should never fail")
}
