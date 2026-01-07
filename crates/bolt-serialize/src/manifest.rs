use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use time::OffsetDateTime;

use crate::format::FormatKind;
use crate::record::RecordMeta;

/// Metadata for a checkpoint (user-provided).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: Option<u64>,
    pub epoch: Option<u64>,
    pub loss: Option<f32>,
    pub custom: HashMap<String, serde_json::Value>,
}

/// Internal manifest tracking all records in a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format_kind: Option<FormatKind>,
    pub records: HashMap<String, RecordMeta>,
    pub shard_count: usize,
    pub created_at: String,
    pub meta: CheckpointMeta,
}

impl CheckpointManifest {
    pub fn new(meta: CheckpointMeta, format_kind: FormatKind) -> Self {
        Self {
            format_kind: Some(format_kind),
            records: HashMap::new(),
            shard_count: 0,
            created_at: OffsetDateTime::now_utc().to_string(),
            meta,
        }
    }

    pub fn add_record(&mut self, meta: RecordMeta) {
        self.records.insert(meta.key.clone(), meta);
    }

    pub fn get_record(&self, key: &str) -> Option<&RecordMeta> {
        self.records.get(key)
    }
}

/// Information about a loaded checkpoint.
#[derive(Clone, Debug)]
pub struct CheckpointInfo {
    pub format_kind: FormatKind,
    pub record_count: usize,
    pub shard_count: usize,
    pub created_at: String,
    pub meta: CheckpointMeta,
}

impl From<&CheckpointManifest> for CheckpointInfo {
    fn from(manifest: &CheckpointManifest) -> Self {
        Self {
            format_kind: manifest
                .format_kind
                .clone()
                .unwrap_or(FormatKind::SafeTensors),
            record_count: manifest.records.len(),
            shard_count: manifest.shard_count,
            created_at: manifest.created_at.clone(),
            meta: manifest.meta.clone(),
        }
    }
}
