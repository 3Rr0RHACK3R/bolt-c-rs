use bolt_core::{backend::{ArgmaxOp, Backend, CopyOp}, dtype::{FloatType, NativeType}, Tensor};

use crate::error::{Error, Result};

pub fn accuracy_top1<B, D>(
    logits: &Tensor<B, D>,
    targets: &Tensor<B, i32>,
) -> Result<f32>
where
    B: Backend + ArgmaxOp<D> + CopyOp<i32>,
    D: FloatType + NativeType,
{
    let logits_shape = logits.shape();
    let target_shape = targets.shape();

    if logits_shape.is_empty() {
        return Err(Error::InvalidTargetShape {
            expected: vec![1],
            got: logits_shape.to_vec(),
        });
    }

    let batch_dims = &logits_shape[..logits_shape.len() - 1];
    if target_shape != batch_dims {
        return Err(Error::InvalidTargetShape {
            expected: batch_dims.to_vec(),
            got: target_shape.to_vec(),
        });
    }

    let preds = logits.argmax(Some(&[-1]), false)?;
    let preds_vec = preds.to_vec()?;
    let targets_vec = targets.to_vec()?;

    if targets_vec.is_empty() {
        return Err(Error::InvalidTargetShape {
            expected: vec![1],
            got: vec![0],
        });
    }

    let correct = preds_vec
        .iter()
        .zip(targets_vec.iter())
        .filter(|(p, t)| p == t)
        .count();

    Ok(correct as f32 / targets_vec.len() as f32)
}

