use bolt_core::dtype::FloatType;

pub trait Float: FloatType + num_traits::Float + Default + Send + Sync {
    fn from_usize(n: usize) -> Self;
}

impl Float for f32 {
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl Float for f64 {
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}
