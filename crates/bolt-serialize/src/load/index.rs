//! Record index building - maps record names to their shard locations.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::format::RecordEntry;
use crate::{Error, RecordMeta, Result};

pub struct RecordIndex {
    pub meta: RecordMeta,
    pub shard_idx: usize,
    pub key: String,
    #[allow(dead_code)]
    pub checksum: Option<String>,
    pub corrupted: bool,
}

pub fn build_record_index(
    record_entries: &BTreeMap<String, RecordEntry>,
    shard_files: &[String],
    dir: &Path,
    corrupted_shards: &[usize],
) -> Result<HashMap<String, RecordIndex>> {
    let shard_path_to_idx: HashMap<&str, usize> = shard_files
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();

    let mut index = HashMap::new();

    for (name, entry) in record_entries {
        let shard_idx = *shard_path_to_idx
            .get(entry.location.shard.as_str())
            .ok_or_else(|| Error::ShardFormat {
                shard: dir.join(&entry.location.shard),
                reason: format!("shard '{}' not found in manifest", entry.location.shard),
            })?;

        let dtype = entry.parse_dtype().ok_or_else(|| Error::ShardFormat {
            shard: dir.join(&entry.location.shard),
            reason: format!("unknown dtype '{}'", entry.dtype),
        })?;

        let corrupted = corrupted_shards.contains(&shard_idx);

        index.insert(
            name.clone(),
            RecordIndex {
                meta: RecordMeta {
                    name: name.clone(),
                    dtype,
                    shape: entry.shape.clone(),
                    role: entry.role.clone(),
                    group: entry.group,
                },
                shard_idx,
                key: entry.location.key.clone(),
                checksum: entry.checksum.clone(),
                corrupted,
            },
        );
    }

    Ok(index)
}
