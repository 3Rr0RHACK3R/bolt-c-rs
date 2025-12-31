//! Checkpoint struct and its query methods.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::load::index::RecordIndex;
use crate::load::shard::LoadedShard;
use crate::{Error, RecordMeta, RecordView, Result, Role};

#[derive(Clone, Debug, Default)]
pub struct CheckpointMeta {
    pub epoch: Option<u64>,
    pub global_step: Option<u64>,
    pub model_name: Option<String>,
    pub user: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct CheckpointInfo {
    pub meta: CheckpointMeta,
    pub written_at: String,
}

pub struct Checkpoint {
    pub(super) info: CheckpointInfo,
    pub(super) dir: PathBuf,
    pub(super) shards: Vec<LoadedShard>,
    pub(super) index: HashMap<String, RecordIndex>,
}

impl Checkpoint {
    pub fn info(&self) -> &CheckpointInfo {
        &self.info
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn list(&self) -> Vec<RecordMeta> {
        let mut out: Vec<_> = self.index.values().map(|idx| idx.meta.clone()).collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    pub fn list_by_role(&self, role: Role) -> Vec<RecordMeta> {
        let mut out: Vec<_> = self
            .index
            .values()
            .filter(|idx| idx.meta.role == role)
            .map(|idx| idx.meta.clone())
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<RecordView<'_>> {
        let idx = self.index.get(name).ok_or_else(|| Error::RecordNotFound {
            name: name.to_string(),
            dir: self.dir.clone(),
        })?;

        if idx.corrupted {
            return Err(Error::RecordUnavailable {
                name: name.to_string(),
            });
        }

        let shard = &self.shards[idx.shard_idx];
        let tensors = shard.tensors()?;
        let tensor_view = tensors.tensor(&idx.key).map_err(|e| Error::ShardFormat {
            shard: shard.path.clone(),
            reason: e.to_string(),
        })?;

        RecordView::from_safetensors_view(&tensor_view, &shard.path)
    }

    pub fn materialize(&self, name: &str) -> Result<Vec<u8>> {
        let view = self.get(name)?;
        Ok(view.data.to_vec())
    }
}
