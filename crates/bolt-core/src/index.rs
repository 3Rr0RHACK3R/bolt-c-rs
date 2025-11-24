use crate::{
    error::{Error, Result},
    layout::TensorIndexer,
};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

/// Trait for individual index elements (e.g. `usize`, `Range`, `RangeFull`).
pub trait TensorIndexElem {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer>;
}

impl TensorIndexElem for usize {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        if *self >= dim {
            return Err(Error::invalid_shape(format!(
                "index {} out of bounds for dimension size {}",
                self, dim
            )));
        }
        Ok(TensorIndexer::Select(*self))
    }
}

impl TensorIndexElem for Range<usize> {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        if self.start > dim || self.end > dim {
            return Err(Error::invalid_shape(format!(
                "range {:?} out of bounds for dimension size {}",
                self, dim
            )));
        }
        Ok(TensorIndexer::Slice {
            start: self.start,
            end: self.end,
            step: 1,
        })
    }
}

impl TensorIndexElem for RangeInclusive<usize> {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        let start = *self.start();
        let end = self
            .end()
            .checked_add(1)
            .ok_or_else(|| Error::invalid_shape("RangeInclusive end overflow"))?;
        if start > dim || end > dim {
            return Err(Error::invalid_shape(format!(
                "range {:?} out of bounds for dimension size {}",
                self, dim
            )));
        }
        Ok(TensorIndexer::Slice {
            start,
            end,
            step: 1,
        })
    }
}

impl TensorIndexElem for RangeFrom<usize> {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        if self.start > dim {
            return Err(Error::invalid_shape(format!(
                "range start {} out of bounds for dimension size {}",
                self.start, dim
            )));
        }
        Ok(TensorIndexer::Slice {
            start: self.start,
            end: dim,
            step: 1,
        })
    }
}

impl TensorIndexElem for RangeTo<usize> {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        if self.end > dim {
            return Err(Error::invalid_shape(format!(
                "range end {} out of bounds for dimension size {}",
                self.end, dim
            )));
        }
        Ok(TensorIndexer::Slice {
            start: 0,
            end: self.end,
            step: 1,
        })
    }
}

impl TensorIndexElem for RangeFull {
    fn to_indexer(&self, dim: usize) -> Result<TensorIndexer> {
        Ok(TensorIndexer::Slice {
            start: 0,
            end: dim,
            step: 1,
        })
    }
}

/// Trait for types that can index a Tensor (producing a list of indexers).
pub trait TensorIndex {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>>;
}

// Implement for single element (auto-wrapped in vec)
impl<T: TensorIndexElem> TensorIndex for T {
    fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        if shape.is_empty() {
            // Scalar tensor indexing? Or 1D tensor indexed by scalar?
            // If T is usize, it expects a dimension.
            // If tensor is scalar (rank 0), shape is empty.
            // Can we index a scalar? `x.i(0)`? No, 0 is OOB for dim size 0?
            // Actually `collect_dims` now allows empty shape.
            // But `to_indexer` takes `dim`. `shape[0]` will panic if empty.
            // So we must check.
            return Err(Error::invalid_shape("cannot index into scalar (rank 0) tensor with an element"));
        }
        Ok(vec![self.to_indexer(shape[0])?])
    }
}

// Implement for collections of pre-computed indexers
impl TensorIndex for Vec<TensorIndexer> {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(self.clone())
    }
}

impl TensorIndex for &[TensorIndexer] {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(self.to_vec())
    }
}

impl<const N: usize> TensorIndex for [TensorIndexer; N] {
    fn to_indexers(&self, _shape: &[usize]) -> Result<Vec<TensorIndexer>> {
        Ok(self.to_vec())
    }
}

/// Implements `TensorIndex` for tuples of `TensorIndexElem`s.
macro_rules! impl_tuple_index {
    ($($T:ident),+) => {
        impl<$($T),+> TensorIndex for ($($T,)+)
        where
            $($T: TensorIndexElem),+
        {
            fn to_indexers(&self, shape: &[usize]) -> Result<Vec<TensorIndexer>> {
                // Calculate tuple arity
                let len = 0 $( + { let _ = stringify!($T); 1 } )+;
                if shape.len() < len {
                     return Err(Error::invalid_shape(format!(
                         "tuple length {} exceeds tensor rank {}", len, shape.len()
                     )));
                }

                let mut indexers = Vec::with_capacity(len);
                #[allow(non_snake_case)]
                let ($($T,)+) = self;
                let mut i = 0;

                $(
                    indexers.push($T.to_indexer(shape[i])?);
                    i += 1;
                )+
                let _ = i;

                Ok(indexers)
            }
        }
    }
}

// Implement for tuples of various arities (up to 12 to match MAX_RANK)
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
