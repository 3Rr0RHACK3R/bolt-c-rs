pub mod abs;
pub mod add;
pub mod add_scalar;
pub mod argmax;
pub mod argmin;
pub mod concat;
pub mod copy;
pub mod cos;
pub mod div;
pub mod div_scalar;
pub mod exp;
pub mod log;
pub mod matmul;
pub mod max;
pub mod mean;
pub mod min;
pub mod mul;
pub mod mul_scalar;
pub mod neg;
pub mod pow;
pub mod prod;
pub mod random;
pub mod reduction_helpers;
pub mod relu;
pub mod sigmoid;
pub mod sin;
pub mod sqrt;
pub mod sub;
pub mod sub_scalar;
pub mod sum;
pub mod tanh;

pub use abs::AbsKernel;
pub use add::AddKernel;
pub use add_scalar::AddScalarKernel;
pub use argmax::ArgmaxKernel;
pub use argmin::ArgminKernel;
pub use concat::ConcatKernel;
pub use copy::CopyKernel;
pub use cos::CosKernel;
pub use div::DivKernel;
pub use div_scalar::DivScalarKernel;
pub use exp::ExpKernel;
pub use log::LogKernel;
pub use matmul::MatmulKernel;
pub use max::MaxKernel;
pub use mean::MeanKernel;
pub use min::MinKernel;
pub use mul::MulKernel;
pub use mul_scalar::MulScalarKernel;
pub use neg::NegKernel;
pub use pow::PowKernel;
pub use prod::ProdKernel;
pub use relu::ReluKernel;
pub use sigmoid::SigmoidKernel;
pub use sin::SinKernel;
pub use sqrt::SqrtKernel;
pub use sub::SubKernel;
pub use sub_scalar::SubScalarKernel;
pub use sum::SumKernel;
pub use tanh::TanhKernel;

use crate::utils::can_use_fast_path_binary;
use bolt_core::dtype::NativeType;

pub trait CpuScalar:
    NativeType
    + Copy
    + Send
    + Sync
    + 'static
    + CopyKernel
    + AddKernel
    + SubKernel
    + MatmulKernel
    + MulKernel
    + NegKernel
    + AbsKernel
    + ReluKernel
    + DivKernel
    + SumKernel
    + ProdKernel
    + MinKernel
    + MaxKernel
    + ArgminKernel
    + ArgmaxKernel
{
}

impl<T> CpuScalar for T where
    T: NativeType
        + Copy
        + Send
        + Sync
        + 'static
        + CopyKernel
        + AddKernel
        + SubKernel
        + MatmulKernel
        + MulKernel
        + NegKernel
        + AbsKernel
        + ReluKernel
        + DivKernel
        + SumKernel
        + ProdKernel
        + MinKernel
        + MaxKernel
        + ArgminKernel
        + ArgmaxKernel
{
}
