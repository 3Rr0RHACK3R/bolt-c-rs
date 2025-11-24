use crate::{
    error::{Error, Result},
    layout::TensorIndexer,
};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

pub trait TensorIndex {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>>;
}

impl TensorIndex for usize {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Select(*self)])
    }
}

impl TensorIndex for Range<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Slice {
            start: self.start,
            end: self.end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeInclusive<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        let start = *self.start();
        let end = self
            .end()
            .checked_add(1)
            .ok_or_else(|| Error::invalid_shape("RangeInclusive end overflow"))?;
        Ok(vec![TensorIndexer::Slice {
            start,
            end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeFrom<usize> {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        if shape.is_empty() {
            return Err(Error::invalid_shape(
                "RangeFrom on 0-rank tensor or indexer mismatch",
            ));
        }
        let dim = shape[0];
        Ok(vec![TensorIndexer::Slice {
            start: self.start,
            end: dim,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeTo<usize> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(vec![TensorIndexer::Slice {
            start: 0,
            end: self.end,
            step: 1,
        }])
    }
}

impl TensorIndex for RangeFull {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        if shape.is_empty() {
             return Ok(vec![]);
        }

        let dim = shape[0];
        Ok(vec![TensorIndexer::Slice {
            start: 0,
            end: dim,
            step: 1,
        }])
    }
}


/// Implements `TensorIndex` for tuples of various sizes (up to arity 6).
/// This macro handles recursive indexing where each tuple element consumes
/// one or more dimensions of the tensor shape.
macro_rules! impl_tuple_index {
    ($($T:ident),+) => {
        impl<$($T),+> TensorIndex for ($($T,)+)
        where
            $($T: TensorIndex),+
        {
            fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
                let mut indexers = Vec::new();
                let mut current_dim = 0;

                #[allow(non_snake_case)]
                let ($($T,)+) = self;

                $(
                    if current_dim < shape.len() {
                        let sub_shape = &shape[current_dim..];
                        let sub_indexers = $T.to_indexers(sub_shape)?;
                        let count = sub_indexers.len();
                        indexers.extend(sub_indexers);
                        current_dim += count;
                    } else {
                        let sub_indexers = $T.to_indexers(&[])?;
                        let count = sub_indexers.len();
                        indexers.extend(sub_indexers);
                        current_dim += count;
                    }
                )+
                // Suppress "unused assignment" warning for the last iteration
                let _ = current_dim;
                Ok(indexers)
            }
        }
    }
}

impl_tuple_index!(A);
impl_tuple_index!(A, B);
impl_tuple_index!(A, B, C);
impl_tuple_index!(A, B, C, D);
impl_tuple_index!(A, B, C, D, E);
impl_tuple_index!(A, B, C, D, E, F);
impl_tuple_index!(A, B, C, D, E, F, G);
impl_tuple_index!(A, B, C, D, E, F, G, H);
impl_tuple_index!(A, B, C, D, E, F, G, H, I);
impl_tuple_index!(A, B, C, D, E, F, G, H, I, J);
impl_tuple_index!(A, B, C, D, E, F, G, H, I, J, K);
impl_tuple_index!(A, B, C, D, E, F, G, H, I, J, K, L);
