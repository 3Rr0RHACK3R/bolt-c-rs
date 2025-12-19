#[derive(Clone, Copy, Debug)]
pub struct IdxSpec {
    pub rows: usize,
    pub cols: usize,
    pub channels: usize,
}

impl IdxSpec {
    pub const fn new(rows: usize, cols: usize, channels: usize) -> Self {
        Self {
            rows,
            cols,
            channels,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IdxSplit {
    Train,
    Test,
}

#[derive(Debug)]
pub struct IdxExample {
    pub pixels: Vec<u8>,
    pub label: u8,
}
