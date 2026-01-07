use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::manifest::CheckpointManifest;
use crate::options::{CheckpointOptions, LoadOpts};
use crate::record::{Record, RecordView};

/// Internal trait for format-specific writers.
pub(crate) trait FormatWriter: Send {
    fn write_record(&mut self, record: Record) -> Result<()>;
    fn flush_shard(&mut self) -> Result<()>;
    fn finish(self: Box<Self>, manifest: &mut CheckpointManifest) -> Result<()>;
}

/// Internal trait for format-specific readers.
pub(crate) trait FormatReader: Send + Sync {
    fn read_record(&self, key: &str) -> Result<RecordView>;
    fn contains(&self, key: &str) -> bool;
    fn list_keys(&self) -> Vec<String>;
}

/// Public enum for format selection.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FormatKind {
    #[default]
    SafeTensors,
    Binary,
}

impl FormatKind {
    /// Create a format writer for this format kind.
    pub(crate) fn create_writer(
        &self,
        dir: &Path,
        opts: &CheckpointOptions,
    ) -> Result<Box<dyn FormatWriter>> {
        match self {
            FormatKind::SafeTensors => Ok(Box::new(safetensors::SafeTensorsFormatWriter::new(
                dir, opts,
            )?)),
            FormatKind::Binary => Ok(Box::new(binary::BinaryFormatWriter::new(dir, opts)?)),
        }
    }

    /// Create a format reader for this format kind.
    pub(crate) fn create_reader(
        &self,
        dir: &Path,
        manifest: &CheckpointManifest,
        opts: &LoadOpts,
    ) -> Result<Box<dyn FormatReader>> {
        match self {
            FormatKind::SafeTensors => Ok(Box::new(safetensors::SafeTensorsFormatReader::new(
                dir, manifest, opts,
            )?)),
            FormatKind::Binary => Ok(Box::new(binary::BinaryFormatReader::new(
                dir, manifest, opts,
            )?)),
        }
    }

    /// Detect format from checkpoint manifest.
    pub fn from_manifest(manifest: &CheckpointManifest) -> Self {
        manifest
            .format_kind
            .clone()
            .unwrap_or(FormatKind::SafeTensors)
    }
}

pub mod binary;
pub mod safetensors;
