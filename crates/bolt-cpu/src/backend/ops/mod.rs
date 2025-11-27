pub mod add;
pub mod copy;
pub mod matmul;
pub mod mean;
pub mod mul;
pub mod sub;

pub use add::AddKernel;
pub use copy::CopyKernel;
pub use matmul::MatmulKernel;
pub use mean::MeanKernel;
pub use mul::MulKernel;
pub use sub::SubKernel;

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
{
}
