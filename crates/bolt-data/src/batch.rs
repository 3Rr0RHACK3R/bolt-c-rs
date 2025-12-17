use std::sync::Arc;

use bolt_core::Backend;

use crate::{Result, Stream};

pub trait BatchFromExamples<B, E>
where
    B: Backend,
{
    fn from_examples(backend: &Arc<B>, examples: &[E]) -> Result<Self>
    where
        Self: Sized;
}

impl<E> Stream<Vec<E>> {
    pub fn to_batches<B, Batch>(self, backend: Arc<B>) -> Stream<Batch>
    where
        B: Backend + 'static,
        Batch: BatchFromExamples<B, E> + Send + 'static,
        E: Send + 'static,
    {
        self.map(move |examples| {
            Batch::from_examples(&backend, &examples).unwrap_or_else(|e| {
                // In a real implementation we might propagate this error
                // via a different API. For now keep it simple and panic
                // loudly so issues are visible.
                panic!("failed to build batch: {e}");
            })
        })
    }
}
