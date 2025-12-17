use crate::{BatchSource, EnumerateSource, MapSource, Result, ShuffleSource, Source, TakeSource};

pub struct Stream<E> {
    pub(crate) inner: Box<dyn Source<E>>, // visible within crate for adapters
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

    pub fn shuffle<R>(self, buffer_size: usize, rng: R) -> Stream<E>
    where
        R: rand::Rng + Clone + Send + 'static,
        E: Send + 'static,
    {
        Stream::new(ShuffleSource::new(self.inner, buffer_size, rng))
    }

    pub fn batch(self, batch_size: usize) -> Stream<Vec<E>>
    where
        E: Send + 'static,
    {
        Stream::new(BatchSource::new(self.inner, batch_size))
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
