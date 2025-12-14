use bolt_core::{
    Tensor,
    backend::{Backend, ExpOp, LogOp, MaxOp, MeanOp, MulOp, NegOp, SubOp, SumOp},
    dtype::Float,
};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

pub fn mse<B, D>(
    pred: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend + SubOp<D> + MulOp<D> + SumOp<D> + MeanOp<D>,
    D: Float,
{
    ensure_same_shape(pred, target)?;

    let diff = pred.sub(target)?;
    let squared = diff.mul(&diff)?;
    apply_reduction(squared, reduction)
}

pub fn cross_entropy_from_logits<B, D>(
    logits: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend
        + SubOp<D>
        + MulOp<D>
        + SumOp<D>
        + MeanOp<D>
        + ExpOp<D>
        + LogOp<D>
        + MaxOp<D>
        + NegOp<D>,
    D: Float,
{
    ensure_same_shape(logits, target)?;

    // log_softmax for numerical stability: logits - logsumexp(logits)
    let max = logits.max(Some(&[-1]), true)?;
    let shifted = logits.sub(&max)?;
    let logsumexp = shifted.exp()?.sum(Some(&[-1]), true)?.log()?;
    let log_probs = shifted.sub(&logsumexp)?;

    let per_sample = target.mul(&log_probs)?.sum(Some(&[-1]), false)?.neg()?;
    apply_reduction(per_sample, reduction)
}

pub fn cross_entropy<B, D>(
    probs: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend + LogOp<D> + MulOp<D> + SumOp<D> + MeanOp<D> + NegOp<D>,
    D: Float,
{
    ensure_same_shape(probs, target)?;
    let log_probs = probs.log()?;
    let per_sample = target.mul(&log_probs)?.sum(Some(&[-1]), false)?.neg()?;
    apply_reduction(per_sample, reduction)
}

fn apply_reduction<B, D>(tensor: Tensor<B, D>, reduction: Reduction) -> Result<Tensor<B, D>>
where
    B: Backend + MeanOp<D> + SumOp<D>,
    D: Float,
{
    match reduction {
        Reduction::None => Ok(tensor),
        Reduction::Mean => Ok(tensor.mean(None, false)?),
        Reduction::Sum => Ok(tensor.sum(None, false)?),
    }
}

fn ensure_same_shape<B, D>(lhs: &Tensor<B, D>, rhs: &Tensor<B, D>) -> Result<()>
where
    B: Backend,
    D: Float,
{
    if lhs.shape() != rhs.shape() {
        return Err(Error::ShapeMismatch {
            expected: lhs.shape().to_vec(),
            got: rhs.shape().to_vec(),
        });
    }
    Ok(())
}
