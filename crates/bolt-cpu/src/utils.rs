use bolt_core::dtype::NativeType;
use bolt_core::layout::Layout;

use crate::backend::CpuTensorView;

#[inline]
pub(crate) fn can_use_fast_path_binary<D: NativeType>(
    lhs: &CpuTensorView<'_, D>,
    rhs: &CpuTensorView<'_, D>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> bool {
    let no_broadcast = lhs_layout.shape() == rhs_layout.shape()
        && lhs_layout.shape() == lhs.layout.shape()
        && rhs_layout.shape() == rhs.layout.shape();

    no_broadcast
        && lhs.layout.is_contiguous()
        && rhs.layout.is_contiguous()
        && lhs.layout.offset_bytes() == 0
        && rhs.layout.offset_bytes() == 0
}
