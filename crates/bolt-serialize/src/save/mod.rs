mod plan;
mod writer;

pub use plan::{ShardEntry, ShardPlan, plan_shards};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use glob::Pattern;

use crate::format::{
    CHECKPOINT_MANIFEST_NAME, CheckpointManifest, CheckpointMetadataJson, RecordEntry,
    RecordLocation, ShardInfo, ensure_shards_dir, write_manifest,
};
use crate::options::SaveOpts;
use crate::{CheckpointMeta, Error, Record, Result};
use writer::write_shard;

pub fn save_checkpoint<'a, I>(
    records: I,
    dir: &Path,
    meta: &CheckpointMeta,
    opts: &SaveOpts,
) -> Result<()>
where
    I: IntoIterator<Item = Result<Record<'a>>>,
{
    let compiled_patterns: Result<Vec<Pattern>> = opts
        .exclude
        .iter()
        .map(|pattern| {
            Pattern::new(pattern).map_err(|e| Error::InvalidExcludePattern {
                pattern: pattern.clone(),
                source: e,
            })
        })
        .collect();
    let compiled_patterns = compiled_patterns?;

    let mut all_records: Vec<Record> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    for result in records {
        let record = result?;

        if compiled_patterns
            .iter()
            .any(|p| p.matches(&record.meta.name))
        {
            continue;
        }

        validate_record_name(&record.meta.name)?;

        let expected_bytes = record.meta.nbytes().ok_or_else(|| Error::NumelOverflow {
            shape: record.meta.shape.as_slice().to_vec(),
        })?;
        let actual_bytes = record.data.len() as u64;
        if expected_bytes != actual_bytes {
            return Err(Error::ByteSizeMismatch {
                name: record.meta.name.clone(),
                expected: expected_bytes,
                actual: actual_bytes,
                numel: record.meta.numel().unwrap_or(0),
                dtype: record.meta.dtype,
            });
        }

        if !seen_names.insert(record.meta.name.clone()) {
            return Err(Error::DuplicateName {
                name: record.meta.name.clone(),
            });
        }

        all_records.push(record);
    }

    if dir.exists() {
        if opts.overwrite {
            fs::remove_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        } else {
            return Err(Error::DirectoryExists {
                dir: dir.to_path_buf(),
            });
        }
    }

    let temp_dir = create_temp_dir(dir, "checkpoint")?;

    match write_checkpoint_to_dir(&all_records, &temp_dir, meta, opts) {
        Ok(()) => {
            fs::rename(&temp_dir, dir).map_err(|e| Error::AtomicRenameFailed {
                temp: temp_dir.clone(),
                final_dir: dir.to_path_buf(),
                source: e,
            })?;
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&temp_dir);
            Err(e)
        }
    }
}

fn write_checkpoint_to_dir(
    records: &[Record],
    dir: &Path,
    meta: &CheckpointMeta,
    opts: &SaveOpts,
) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    ensure_shards_dir(dir)?;

    let shard_plans = plan_shards(records, opts.shard_max_bytes);

    let mut shard_checksums = Vec::new();
    let mut all_record_checksums: HashMap<String, String> = HashMap::new();

    for plan in &shard_plans {
        let shard_path = dir.join(plan.relative_path());
        let result = write_shard(plan, records, &shard_path, opts.alignment, opts.checksum)?;

        if opts.checksum {
            shard_checksums.push(result.shard_checksum);
            all_record_checksums.extend(result.record_checksums);
        }
    }

    let mut manifest = CheckpointManifest::new();
    manifest.metadata = CheckpointMetadataJson {
        epoch: meta.epoch,
        global_step: meta.global_step,
        model_name: meta.model_name.clone(),
        user: meta.user.clone(),
    };
    manifest.shards = ShardInfo {
        files: shard_plans.iter().map(|s| s.relative_path()).collect(),
        checksums: shard_checksums,
    };

    for plan in &shard_plans {
        for entry in &plan.entries {
            let record = &records[entry.record_idx];
            let record_checksum = all_record_checksums.get(&record.meta.name).cloned();

            manifest.tensors.insert(
                record.meta.name.clone(),
                RecordEntry {
                    role: record.meta.role.clone(),
                    group: record.meta.group,
                    dtype: record.meta.dtype.name().to_string(),
                    shape: record.meta.shape.clone(),
                    location: RecordLocation {
                        shard: plan.relative_path(),
                        key: record.meta.name.clone(),
                    },
                    checksum: record_checksum,
                },
            );
        }
    }

    let manifest_path = dir.join(CHECKPOINT_MANIFEST_NAME);
    write_manifest(&manifest, &manifest_path)?;

    Ok(())
}

fn validate_record_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot be empty".to_string(),
        });
    }

    if name.contains('\0') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot contain NUL character".to_string(),
        });
    }

    if name.contains('/') || name.contains('\\') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot contain path separators (/ or \\)".to_string(),
        });
    }

    if name == ".." {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot be '..' (reserved directory name)".to_string(),
        });
    }

    if name.starts_with('.') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot start with '.'".to_string(),
        });
    }

    Ok(())
}

fn create_temp_dir(target: &Path, default_name: &str) -> Result<std::path::PathBuf> {
    let parent = target.parent().unwrap_or(Path::new("."));
    let name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(default_name);
    let uuid = uuid::Uuid::new_v4();
    let temp_name = format!("{}.tmp-{}", name, uuid);
    let temp_path = parent.join(temp_name);

    fs::create_dir_all(&temp_path).map_err(|e| Error::io(&temp_path, e))?;

    Ok(temp_path)
}
