pub mod abs;
pub mod add;
pub mod copy;
pub mod cos;
pub mod exp;
pub mod log;
pub mod matmul;
pub mod mean;
pub mod mul;
pub mod neg;
pub mod relu;
pub mod sin;
pub mod sqrt;
pub mod sub;
pub mod tanh;

pub use abs::AbsKernel;
pub use add::AddKernel;
pub use copy::CopyKernel;
pub use cos::CosKernel;
pub use exp::ExpKernel;
pub use log::LogKernel;
pub use matmul::MatmulKernel;
pub use mean::MeanKernel;
pub use mul::MulKernel;
pub use neg::NegKernel;
pub use relu::ReluKernel;
pub use sin::SinKernel;
pub use sqrt::SqrtKernel;
pub use sub::SubKernel;
pub use tanh::TanhKernel;

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
    + MeanKernel
    + NegKernel
    + AbsKernel
    + ReluKernel
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
        + MeanKernel
        + NegKernel
        + AbsKernel
        + ReluKernel
{
}
