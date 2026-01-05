#![deny(unused_must_use)]

mod error;
mod manifest;
mod options;
mod reader;
mod record;
mod traits;
mod writer;

pub mod adapters;
pub mod format;

pub use error::{Error, Result};
pub use manifest::{CheckpointInfo, CheckpointManifest, CheckpointMeta};
pub use options::{CheckpointOptions, FormatKind, LoadOpts};
pub use reader::CheckpointReader;
pub use traits::{LoadCheckpoint, SaveCheckpoint};
pub use writer::CheckpointWriter;

/// Save a single item implementing `SaveCheckpoint`.
pub fn save<T: SaveCheckpoint>(
    item: &T,
    dir: &std::path::Path,
    meta: &CheckpointMeta,
    opts: &CheckpointOptions,
) -> Result<()> {
    let mut writer = CheckpointWriter::new(dir, opts)?;
    item.save(&mut writer)?;
    writer.finish(meta)
}

/// Save multiple items with prefixes.
/// This function is object-safe and can accept heterogeneous collections via `&dyn SaveCheckpoint`.
pub fn save_all(
    items: &[(&str, &dyn SaveCheckpoint)],
    dir: &std::path::Path,
    meta: &CheckpointMeta,
    opts: &CheckpointOptions,
) -> Result<()> {
    let mut writer = CheckpointWriter::new(dir, opts)?;
    for (prefix, item) in items {
        writer.save_prefixed(prefix, *item)?;
    }
    writer.finish(meta)
}

/// Load a single item implementing `LoadCheckpoint`.
pub fn load<T: LoadCheckpoint>(
    item: &mut T,
    dir: &std::path::Path,
    opts: &LoadOpts,
) -> Result<CheckpointInfo> {
    let mut reader = CheckpointReader::open(dir, opts)?;
    item.load(&mut reader)?;
    Ok(reader.info().clone())
}

/// Load multiple items with prefixes.
/// This function is object-safe and can accept heterogeneous collections via `&mut dyn LoadCheckpoint`.
pub fn load_all(
    items: &mut [(&str, &mut dyn LoadCheckpoint)],
    dir: &std::path::Path,
    opts: &LoadOpts,
) -> Result<CheckpointInfo> {
    let mut reader = CheckpointReader::open(dir, opts)?;
    for (prefix, item) in items {
        reader.load_prefixed(prefix, *item)?;
    }
    Ok(reader.info().clone())
}
