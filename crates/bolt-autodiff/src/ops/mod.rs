mod abs;
mod add;
mod cos;
mod div;
mod exp;
mod expand;
mod log;
mod matmul;
mod mean;
mod mul;
mod neg;
mod pow;
mod relu;
mod reshape;
mod sin;
mod sqrt;
mod squeeze;
mod sub;
mod sum;
mod tanh;
mod transpose;
mod unsqueeze;

pub use abs::AbsBackward;
pub use add::AddBackward;
pub use cos::CosBackward;
pub use div::DivBackward;
pub use exp::ExpBackward;
pub use expand::ExpandBackward;
pub use log::LogBackward;
pub use matmul::MatmulBackward;
pub use mean::MeanBackward;
pub use mul::MulBackward;
pub use neg::NegBackward;
pub use pow::PowBackward;
pub use relu::ReluBackward;
pub use reshape::ReshapeBackward;
pub use sin::SinBackward;
pub use sqrt::SqrtBackward;
pub use squeeze::SqueezeBackward;
pub use sub::SubBackward;
pub use sum::SumBackward;
pub use tanh::TanhBackward;
pub use transpose::TransposeBackward;
pub use unsqueeze::UnsqueezeBackward;

use bolt_core::backend::{AddOp, SumOp};
use bolt_core::{Backend, Tensor};

use crate::Float;
use crate::error::Result;

pub(crate) fn reduce_grad_to_shape<B, D>(
    grad: &Tensor<B, D>,
    target_shape: &[usize],
) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + SumOp<D>,
    D: Float,
{
    let grad_shape = grad.shape();

    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    if target_shape.is_empty() {
        return Ok(grad.sum(None, false)?);
    }

    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    if grad_rank < target_rank {
        return Err(crate::error::Error::Core(bolt_core::Error::ShapeMismatch {
            lhs: grad_shape.to_vec(),
            rhs: target_shape.to_vec(),
        }));
    }

    let rank_diff = grad_rank - target_rank;
    let mut result = grad.clone();

    for i in 0..rank_diff {
        result = result.sum(Some(&[0]), false)?;
        let _ = i;
    }

    for i in 0..target_rank {
        if target_shape[i] == 1 && result.shape()[i] != 1 {
            result = result.sum(Some(&[i as isize]), true)?;
        }
    }

    Ok(result)
}
