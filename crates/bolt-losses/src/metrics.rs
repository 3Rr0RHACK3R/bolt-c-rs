use bolt_core::{
    backend::{Backend, CopyOp},
    dtype::Float,
};
use bolt_tensor::Tensor;

use crate::error::{Error, Result};

/// Computes top-k accuracy: fraction of samples where the target class is in the top-k predictions.
pub fn accuracy_topk<B, D>(logits: &Tensor<B, D>, targets: &Tensor<B, i32>, k: usize) -> Result<f32>
where
    B: Backend + CopyOp<D> + CopyOp<i32>,
    D: Float + PartialOrd,
{
    let logits_shape = logits.shape();
    let target_shape = targets.shape();

    if logits_shape.is_empty() {
        return Err(Error::InvalidTargetShape {
            expected: vec![1],
            got: logits_shape.to_vec(),
        });
    }

    let batch_dims = &logits_shape.as_slice()[..logits_shape.len() - 1];
    if target_shape.as_slice() != batch_dims {
        return Err(Error::InvalidTargetShape {
            expected: batch_dims.to_vec(),
            got: target_shape.to_vec(),
        });
    }

    let num_classes = logits_shape.as_slice()[logits_shape.len() - 1];
    if k == 0 || k > num_classes {
        return Err(Error::Core(bolt_core::Error::OpError(format!(
            "k must be between 1 and {}, got {}",
            num_classes, k
        ))));
    }

    let logits_vec = logits.to_vec()?;
    let targets_vec = targets.to_vec()?;

    if targets_vec.is_empty() {
        return Err(Error::InvalidTargetShape {
            expected: vec![1],
            got: vec![0],
        });
    }

    let batch_size = targets_vec.len();
    let mut correct = 0;

    for (sample_idx, &target) in targets_vec.iter().enumerate() {
        let start_idx = sample_idx * num_classes;
        let end_idx = start_idx + num_classes;
        let sample_logits = &logits_vec[start_idx..end_idx];

        // Create (value, index) pairs and sort by value descending
        let mut indexed: Vec<(D, usize)> = sample_logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (val, idx))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Check if target is in top-k
        let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(_, idx)| *idx).collect();
        if top_k_indices.contains(&(target as usize)) {
            correct += 1;
        }
    }

    Ok(correct as f32 / batch_size as f32)
}

pub fn accuracy_top1<B, D>(logits: &Tensor<B, D>, targets: &Tensor<B, i32>) -> Result<f32>
where
    B: Backend + CopyOp<D> + CopyOp<i32>,
    D: Float + PartialOrd,
{
    accuracy_topk(logits, targets, 1)
}
