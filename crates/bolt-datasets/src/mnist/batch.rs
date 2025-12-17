use std::sync::Arc;

use bolt_core::Tensor;
use bolt_core::backend::Backend;
use bolt_data::{BatchFromExamples, DataError, IdxExample};

use super::NUM_CLASSES;

pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, f32>,
    pub targets: Tensor<B, f32>,
    pub labels: Tensor<B, i32>,
}

impl<B: Backend> BatchFromExamples<B, IdxExample> for MnistBatch<B> {
    fn from_examples(backend: &Arc<B>, examples: &[IdxExample]) -> bolt_data::Result<Self> {
        if examples.is_empty() {
            return Err(DataError::EmptyBatch);
        }

        let n = examples.len();
        let pix_per = examples[0].pixels.len();
        if pix_per == 0 {
            return Err(DataError::InvalidShape(
                "idx example has empty pixel buffer".to_string(),
            ));
        }

        let mut img_buf = Vec::with_capacity(n * pix_per);
        let mut label_buf = Vec::with_capacity(n);
        let mut target_buf = vec![0f32; n * NUM_CLASSES];

        for (i, ex) in examples.iter().enumerate() {
            if ex.pixels.len() != pix_per {
                return Err(DataError::InvalidShape(format!(
                    "expected {} pixels, got {}",
                    pix_per,
                    ex.pixels.len()
                )));
            }

            for &p in &ex.pixels {
                img_buf.push(p as f32 / 255.0);
            }

            let label = ex.label as usize;
            if label >= NUM_CLASSES {
                return Err(DataError::InvalidShape(format!(
                    "label {} out of range [0, {})",
                    label, NUM_CLASSES
                )));
            }
            label_buf.push(label as i32);
            target_buf[i * NUM_CLASSES + label] = 1.0;
        }

        let images = Tensor::from_slice(backend, &img_buf, &[n, pix_per])
            .map_err(|e| DataError::Batch(format!("failed to build images tensor: {e}")))?;
        let targets = Tensor::from_slice(backend, &target_buf, &[n, NUM_CLASSES])
            .map_err(|e| DataError::Batch(format!("failed to build targets tensor: {e}")))?;
        let labels = Tensor::from_slice(backend, &label_buf, &[n])
            .map_err(|e| DataError::Batch(format!("failed to build labels tensor: {e}")))?;

        Ok(Self {
            images,
            targets,
            labels,
        })
    }
}
