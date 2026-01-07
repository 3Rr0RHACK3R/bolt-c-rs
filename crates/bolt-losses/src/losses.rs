use bolt_core::{
    backend::{
        AbsOp, AddOp, Backend, CopyOp, ExpOp, FillOp, LogOp, MaxOp, MeanOp, MulOp, NegOp, ReluOp,
        ReshapeOp, SubOp, SumOp,
    },
    dtype::Float,
};
use bolt_tensor::Tensor;

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
    B: Backend + CopyOp<D> + SubOp<D> + MulOp<D> + SumOp<D> + MeanOp<D> + NegOp<D> + ReshapeOp<D>,
    D: Float + PartialEq + PartialOrd,
{
    ensure_same_shape(pred, target)?;

    let diff = pred.sub(target)?;
    let squared = diff.mul(&diff)?;
    apply_reduction(squared, reduction)
}

pub fn mae<B, D>(
    pred: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend + CopyOp<D> + SubOp<D> + AbsOp<D> + SumOp<D> + MeanOp<D> + ReshapeOp<D> + NegOp<D>,
    D: Float + PartialEq + PartialOrd,
{
    ensure_same_shape(pred, target)?;

    let diff = pred.sub(target)?;
    let abs_diff = diff.abs()?;
    apply_reduction(abs_diff, reduction)
}

pub fn cross_entropy_from_logits<B, D>(
    logits: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend
        + CopyOp<D>
        + SubOp<D>
        + MulOp<D>
        + SumOp<D>
        + MeanOp<D>
        + ExpOp<D>
        + LogOp<D>
        + MaxOp<D>
        + NegOp<D>
        + ReshapeOp<D>
        + SumOp<D>,
    D: Float + PartialEq + PartialOrd,
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
    B: Backend + CopyOp<D> + LogOp<D> + MulOp<D> + SumOp<D> + MeanOp<D> + NegOp<D> + ReshapeOp<D>,
    D: Float,
{
    ensure_same_shape(probs, target)?;
    let log_probs = probs.log()?;
    let per_sample = target.mul(&log_probs)?.sum(Some(&[-1]), false)?.neg()?;
    apply_reduction(per_sample, reduction)
}

pub fn cross_entropy_from_logits_sparse<B, D>(
    logits: &Tensor<B, D>,
    targets: &Tensor<B, i32>,
    num_classes: usize,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend
        + CopyOp<D>
        + SubOp<D>
        + MulOp<D>
        + SumOp<D>
        + MeanOp<D>
        + ExpOp<D>
        + LogOp<D>
        + MaxOp<D>
        + NegOp<D>
        + CopyOp<i32>
        + ReshapeOp<D>,
    D: Float + PartialEq + PartialOrd,
{
    let target_tensor: Tensor<B, D> =
        Tensor::<B, D>::one_hot(targets, num_classes).map_err(Error::Core)?;

    cross_entropy_from_logits(logits, &target_tensor, reduction)
}

fn apply_reduction<B, D>(tensor: Tensor<B, D>, reduction: Reduction) -> Result<Tensor<B, D>>
where
    B: Backend + CopyOp<D> + MeanOp<D> + SumOp<D>,
    D: Float,
{
    match reduction {
        Reduction::None => Ok(tensor),
        Reduction::Mean => Ok(tensor.mean(None, false)?),
        Reduction::Sum => Ok(tensor.sum(None, false)?),
    }
}

pub fn binary_cross_entropy<B, D>(
    pred: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend
        + CopyOp<D>
        + AddOp<D>
        + SubOp<D>
        + MulOp<D>
        + LogOp<D>
        + SumOp<D>
        + MeanOp<D>
        + NegOp<D>
        + FillOp<D>
        + ReshapeOp<D>,
    D: Float + PartialEq + PartialOrd,
{
    ensure_same_shape(pred, target)?;

    let eps = D::from_f64(1e-8);
    let ones = Tensor::ones_like(pred)?;
    let eps_tensor = Tensor::full_like(pred, eps)?;
    let pred_clamped = pred.add(&eps_tensor)?;
    let one_minus_pred = ones.sub(pred)?;
    let one_minus_pred_clamped = one_minus_pred.add(&eps_tensor)?;

    let term1 = target.mul(&pred_clamped.log()?)?;
    let term2 = ones.sub(target)?.mul(&one_minus_pred_clamped.log()?)?;
    let loss = term1.add(&term2)?.neg()?;

    apply_reduction(loss, reduction)
}

pub fn binary_cross_entropy_with_logits<B, D>(
    logits: &Tensor<B, D>,
    target: &Tensor<B, D>,
    reduction: Reduction,
) -> Result<Tensor<B, D>>
where
    B: Backend
        + CopyOp<D>
        + AddOp<D>
        + SubOp<D>
        + MulOp<D>
        + ExpOp<D>
        + LogOp<D>
        + AbsOp<D>
        + ReluOp<D>
        + SumOp<D>
        + MeanOp<D>
        + NegOp<D>
        + FillOp<D>
        + ReshapeOp<D>,
    D: Float + PartialEq + PartialOrd,
{
    ensure_same_shape(logits, target)?;

    let max_logits = logits.relu()?;
    let abs_logits = logits.abs()?;
    let neg_abs_logits = abs_logits.neg()?;
    let exp_term = neg_abs_logits.exp()?;
    let ones = Tensor::ones_like(logits)?;
    let log_term = ones.add(&exp_term)?.log()?;

    let loss = max_logits.sub(&logits.mul(target)?)?.add(&log_term)?;

    apply_reduction(loss, reduction)
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
