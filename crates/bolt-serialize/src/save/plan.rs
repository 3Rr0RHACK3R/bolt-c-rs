use crate::format::SHARDS_DIR;
use crate::Record;

#[derive(Clone, Debug)]
pub struct ShardPlan {
    pub index: usize,
    pub total: usize,
    pub entries: Vec<ShardEntry>,
}

#[derive(Clone, Debug)]
pub struct ShardEntry {
    pub record_idx: usize,
    pub start: u64,
    pub end: u64,
}

impl ShardPlan {
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

const DEFAULT_SHARD_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn plan_shards(records: &[Record], max_bytes: Option<u64>) -> Vec<ShardPlan> {
    let max_bytes = max_bytes.unwrap_or(DEFAULT_SHARD_MAX_BYTES);

    let mut sorted_indices: Vec<usize> = (0..records.len()).collect();
    sorted_indices.sort_by(|&a, &b| records[a].meta.name.cmp(&records[b].meta.name));

    let mut shards: Vec<ShardPlan> = Vec::new();
    let mut current_entries: Vec<ShardEntry> = Vec::new();
    let mut current_offset: u64 = 0;

    for &record_idx in &sorted_indices {
        let record = &records[record_idx];
        let record_bytes = record.data.len() as u64;

        if record_bytes > max_bytes {
            if !current_entries.is_empty() {
                shards.push(ShardPlan {
                    index: shards.len(),
                    total: 0,
                    entries: std::mem::take(&mut current_entries),
                });
                current_offset = 0;
            }
            shards.push(ShardPlan {
                index: shards.len(),
                total: 0,
                entries: vec![ShardEntry {
                    record_idx,
                    start: 0,
                    end: record_bytes,
                }],
            });
            continue;
        }

        if current_offset + record_bytes > max_bytes && !current_entries.is_empty() {
            shards.push(ShardPlan {
                index: shards.len(),
                total: 0,
                entries: std::mem::take(&mut current_entries),
            });
            current_offset = 0;
        }

        current_entries.push(ShardEntry {
            record_idx,
            start: current_offset,
            end: current_offset + record_bytes,
        });
        current_offset += record_bytes;
    }

    if !current_entries.is_empty() {
        shards.push(ShardPlan {
            index: shards.len(),
            total: 0,
            entries: current_entries,
        });
    }

    let total = shards.len();
    for shard in &mut shards {
        shard.total = total;
    }

    shards
}
