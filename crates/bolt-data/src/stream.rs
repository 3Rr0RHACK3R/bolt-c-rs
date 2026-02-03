use crate::{BatchSource, EnumerateSource, MapSource, Result, ShuffleSource, Source, TakeSource};
use bolt_rng::RngKey;

pub struct Stream<E> {
    pub(crate) inner: Box<dyn Source<E>>, // visible within crate for adapters
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchRemainder {
    Keep,
    DropLast,
}

impl<E> Stream<E> {
    pub fn new<S>(source: S) -> Self
    where
        S: Source<E> + 'static,
    {
        Self {
            inner: Box::new(source),
        }
    }

    pub fn map<F, E2>(self, f: F) -> Stream<E2>
    where
        F: Fn(E) -> E2 + Send + Sync + 'static,
        E: Send + 'static,
        E2: Send + 'static,
    {
        Stream::new(MapSource {
            inner: self.inner,
            f,
        })
    }

    pub fn shuffle(self, buffer_size: usize, key: RngKey) -> Stream<E>
    where
        E: Send + 'static,
    {
        Stream::new(ShuffleSource::new(self.inner, buffer_size, key))
    }

    pub fn batch(self, batch_size: usize) -> Stream<Vec<E>>
    where
        E: Send + 'static,
    {
        self.batch_with(batch_size, BatchRemainder::Keep)
    }

    pub fn batch_with(self, batch_size: usize, remainder: BatchRemainder) -> Stream<Vec<E>>
    where
        E: Send + 'static,
    {
        let source = match remainder {
            BatchRemainder::Keep => BatchSource::new(self.inner, batch_size),
            BatchRemainder::DropLast => BatchSource::new_drop_last(self.inner, batch_size),
        };
        Stream::new(source)
    }

    pub fn take(self, n: usize) -> Stream<E>
    where
        E: Send + 'static,
    {
        Stream::new(TakeSource::new(self.inner, n))
    }

    pub fn enumerate(self) -> Stream<(usize, E)>
    where
        E: Send + 'static,
    {
        Stream::new(EnumerateSource::new(self.inner))
    }

    pub fn iter(self) -> StreamIter<E> {
        StreamIter { inner: self.inner }
    }
}

pub struct StreamIter<E> {
    pub(crate) inner: Box<dyn Source<E>>,
}

impl<E> Iterator for StreamIter<E> {
    type Item = Result<E>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Ok(Some(e)) => Some(Ok(e)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
