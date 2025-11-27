use crate::layout::Layout;

/// Borrowed view passed into kernels (non-owning).
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, S> {
    pub storage: &'a S,
    pub layout: &'a Layout,
}

impl<'a, S> TensorView<'a, S> {
    pub fn new(storage: &'a S, layout: &'a Layout) -> Self {
        Self { storage, layout }
    }
}
