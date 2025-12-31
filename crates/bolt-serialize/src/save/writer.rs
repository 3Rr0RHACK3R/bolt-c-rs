use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use bolt_core::DType;
use serde::Serialize;

use crate::save::plan::ShardPlan;
use crate::{Error, Record, Result};

pub struct ShardWriteResult {
    pub shard_checksum: String,
    pub record_checksums: HashMap<String, String>,
}

#[derive(Serialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub fn write_shard(
    plan: &ShardPlan,
    records: &[Record],
    out_path: &Path,
    alignment_bytes: u64,
    compute_checksums: bool,
) -> Result<ShardWriteResult> {
    let mut header_map: BTreeMap<String, TensorInfo> = BTreeMap::new();
    for entry in &plan.entries {
        let record = &records[entry.record_idx];
        header_map.insert(
            record.meta.name.clone(),
            TensorInfo {
                dtype: safetensors_dtype_string(record.meta.dtype),
                shape: record.meta.shape.as_slice().to_vec(),
                data_offsets: [entry.start, entry.end],
            },
        );
    }

    let header_json = serde_json::to_string(&header_map).map_err(|e| Error::ShardFormat {
        shard: out_path.to_path_buf(),
        reason: format!("failed to serialize header: {e}"),
    })?;

    let header_bytes = header_json.as_bytes();
    let base_padding = (8 - (header_bytes.len() % 8)) % 8;

    let data_start = 8 + header_bytes.len() + base_padding;
    let extra_padding = if alignment_bytes > 8 {
        let remainder = data_start as u64 % alignment_bytes;
        if remainder == 0 {
            0
        } else {
            (alignment_bytes - remainder) as usize
        }
    } else {
        0
    };

    let total_padding = base_padding + extra_padding;
    let padded_header_len = header_bytes.len() + total_padding;

    let file = File::create(out_path).map_err(|e| Error::io(out_path, e))?;
    let mut writer = BufWriter::new(file);
    let mut shard_hasher = compute_checksums.then(blake3::Hasher::new);
    let mut record_checksums = HashMap::new();

    let header_len_bytes = (padded_header_len as u64).to_le_bytes();
    writer
        .write_all(&header_len_bytes)
        .map_err(|e| Error::io(out_path, e))?;
    if let Some(h) = &mut shard_hasher {
        h.update(&header_len_bytes);
    }

    writer
        .write_all(header_bytes)
        .map_err(|e| Error::io(out_path, e))?;
    if let Some(h) = &mut shard_hasher {
        h.update(header_bytes);
    }

    if total_padding > 0 {
        let padding = vec![b' '; total_padding];
        writer
            .write_all(&padding)
            .map_err(|e| Error::io(out_path, e))?;
        if let Some(h) = &mut shard_hasher {
            h.update(&padding);
        }
    }

    for entry in &plan.entries {
        let record = &records[entry.record_idx];
        let data = record.data.as_ref();

        writer.write_all(data).map_err(|e| Error::io(out_path, e))?;

        if let Some(h) = &mut shard_hasher {
            h.update(data);
        }

        if compute_checksums {
            let hash = blake3::hash(data);
            record_checksums.insert(record.meta.name.clone(), format!("b3:{}", hash.to_hex()));
        }
    }

    writer.flush().map_err(|e| Error::io(out_path, e))?;

    let shard_checksum = shard_hasher
        .map(|h| format!("b3:{}", h.finalize().to_hex()))
        .unwrap_or_default();

    Ok(ShardWriteResult {
        shard_checksum,
        record_checksums,
    })
}

fn safetensors_dtype_string(dtype: DType) -> String {
    match dtype {
        DType::U8 => "U8",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F32 => "F32",
        DType::F64 => "F64",
    }
    .to_string()
}
