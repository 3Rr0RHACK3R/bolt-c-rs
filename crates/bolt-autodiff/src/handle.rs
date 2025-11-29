#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Handle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl Handle {
    pub const NONE: Handle = Handle {
        index: u32::MAX,
        generation: 0,
    };

    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn is_none(&self) -> bool {
        self.index == u32::MAX
    }
}
