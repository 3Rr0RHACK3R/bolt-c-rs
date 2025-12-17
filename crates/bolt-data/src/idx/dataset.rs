
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use crate::Source;
use crate::{DataError, Result};

use super::header::{read_images_header, read_labels_header};
use super::types::{IdxExample, IdxSpec};

#[derive(Debug)]
pub struct IdxDataset {
    pub spec: IdxSpec,
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub images_path: String,
    pub labels_path: String,
}

impl IdxDataset {
    pub fn from_paths(
        spec: IdxSpec,
        images_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let images_path_ref = images_path.as_ref();
        let labels_path_ref = labels_path.as_ref();

        let (meta, images_file) = read_images_header(images_path_ref)?;
        let (label_count, labels_file) = read_labels_header(labels_path_ref)?;

        if meta.n != label_count {
            return Err(DataError::LengthMismatch {
                expected: meta.n,
                actual: label_count,
            });
        }

        if spec.rows != meta.rows || spec.cols != meta.cols {
            return Err(DataError::InvalidShape(format!(
                "spec rows/cols ({}/{}) do not match IDX header ({}/{})",
                spec.rows, spec.cols, meta.rows, meta.cols
            )));
        }

        let _ = images_file;
        let _ = labels_file;

        Ok(Self {
            spec,
            n: meta.n,
            rows: meta.rows,
            cols: meta.cols,
            images_path: images_path_ref.to_string_lossy().into_owned(),
            labels_path: labels_path_ref.to_string_lossy().into_owned(),
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }
}

pub struct IdxSource {
    dataset: Arc<IdxDataset>,
    idx: usize,
    images: File,
    labels: File,
}

impl IdxSource {
    pub fn new(dataset: Arc<IdxDataset>) -> Result<Self> {
        let images = File::open(&dataset.images_path)?;
        let labels = File::open(&dataset.labels_path)?;

        let mut images = images;
        let mut labels = labels;

        images.seek(SeekFrom::Start(16))?;
        labels.seek(SeekFrom::Start(8))?;

        Ok(Self {
            dataset,
            idx: 0,
            images,
            labels,
        })
    }
}

impl Source<IdxExample> for IdxSource {
    fn next(&mut self) -> Result<Option<IdxExample>> {
        if self.idx >= self.dataset.len() {
            return Ok(None);
        }

        let pix_per = self
            .dataset
            .rows
            .checked_mul(self.dataset.cols)
            .and_then(|v| v.checked_mul(self.dataset.spec.channels))
            .ok_or_else(|| DataError::InvalidShape("pixel count overflow".to_string()))?;

        let mut buf = vec![0u8; pix_per];
        self.images.read_exact(&mut buf)?;

        let mut label_buf = [0u8; 1];
        self.labels.read_exact(&mut label_buf)?;
        let label = label_buf[0];

        self.idx += 1;

        Ok(Some(IdxExample { pixels: buf, label }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len().saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}
