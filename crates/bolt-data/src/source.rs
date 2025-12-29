use crate::{DataError, Result};
use bolt_rng::RngStream;

pub trait Source<E>: Send {
    fn next(&mut self) -> Result<Option<E>>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

pub struct MapSource<E, E2, F>
where
    F: Fn(E) -> E2 + Send + Sync + 'static,
{
    pub(crate) inner: Box<dyn Source<E>>,
    pub(crate) f: F,
}

impl<E, E2, F> Source<E2> for MapSource<E, E2, F>
where
    F: Fn(E) -> E2 + Send + Sync + 'static,
{
    fn next(&mut self) -> Result<Option<E2>> {
        match self.inner.next()? {
            Some(e) => Ok(Some((self.f)(e))),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct TryMapSource<E, E2, F>
where
    F: Fn(E) -> Result<E2> + Send + Sync + 'static,
{
    pub(crate) inner: Box<dyn Source<E>>,
    pub(crate) f: F,
}

impl<E, E2, F> Source<E2> for TryMapSource<E, E2, F>
where
    F: Fn(E) -> Result<E2> + Send + Sync + 'static,
{
    fn next(&mut self) -> Result<Option<E2>> {
        match self.inner.next()? {
            Some(e) => Ok(Some((self.f)(e)?)),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct MapWithSource<C, E, E2, F>
where
    C: Send + 'static,
    F: Fn(&C, E) -> E2 + Send + Sync + 'static,
{
    pub(crate) inner: Box<dyn Source<E>>,
    pub(crate) ctx: C,
    pub(crate) f: F,
}

impl<C, E, E2, F> Source<E2> for MapWithSource<C, E, E2, F>
where
    C: Send + 'static,
    F: Fn(&C, E) -> E2 + Send + Sync + 'static,
{
    fn next(&mut self) -> Result<Option<E2>> {
        match self.inner.next()? {
            Some(e) => Ok(Some((self.f)(&self.ctx, e))),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct TryMapWithSource<C, E, E2, F>
where
    C: Send + 'static,
    F: Fn(&C, E) -> Result<E2> + Send + Sync + 'static,
{
    pub(crate) inner: Box<dyn Source<E>>,
    pub(crate) ctx: C,
    pub(crate) f: F,
}

impl<C, E, E2, F> Source<E2> for TryMapWithSource<C, E, E2, F>
where
    C: Send + 'static,
    F: Fn(&C, E) -> Result<E2> + Send + Sync + 'static,
{
    fn next(&mut self) -> Result<Option<E2>> {
        match self.inner.next()? {
            Some(e) => Ok(Some((self.f)(&self.ctx, e)?)),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct ShuffleSource<E> {
    inner: Box<dyn Source<E>>,
    buffer: Vec<E>,
    buffer_size: usize,
    rng: RngStream,
    upstream_finished: bool,
}

impl<E> ShuffleSource<E> {
    pub fn new(inner: Box<dyn Source<E>>, buffer_size: usize, rng: RngStream) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            rng,
            upstream_finished: false,
        }
    }

    fn fill_buffer(&mut self) -> Result<()> {
        while self.buffer.len() < self.buffer_size && !self.upstream_finished {
            match self.inner.next()? {
                Some(e) => self.buffer.push(e),
                None => {
                    self.upstream_finished = true;
                    break;
                }
            }
        }
        Ok(())
    }
}

impl<E> Source<E> for ShuffleSource<E>
where
    E: Send,
{
    fn next(&mut self) -> Result<Option<E>> {
        if self.buffer_size == 0 {
            return Err(DataError::InvalidShape(
                "shuffle buffer_size must be > 0".to_string(),
            ));
        }

        self.fill_buffer()?;

        if self.buffer.is_empty() {
            return Ok(None);
        }

        let len = self.buffer.len();
        let idx = if len == 1 {
            0
        } else {
            self.rng.gen_range(0..len)
        };

        let e = self.buffer.swap_remove(idx);
        Ok(Some(e))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We cannot give precise hints once shuffling
        (0, None)
    }
}

pub struct BatchSource<E> {
    inner: Box<dyn Source<E>>,
    batch_size: usize,
}

impl<E> BatchSource<E> {
    pub fn new(inner: Box<dyn Source<E>>, batch_size: usize) -> Self {
        Self { inner, batch_size }
    }
}

impl<E> Source<Vec<E>> for BatchSource<E> {
    fn next(&mut self) -> Result<Option<Vec<E>>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        while batch.len() < self.batch_size {
            match self.inner.next()? {
                Some(e) => batch.push(e),
                None => break,
            }
        }

        if batch.is_empty() {
            return Ok(None);
        }

        Ok(Some(batch))
    }
}

pub struct TakeSource<E> {
    inner: Box<dyn Source<E>>,
    remaining: usize,
}

impl<E> TakeSource<E> {
    pub fn new(inner: Box<dyn Source<E>>, n: usize) -> Self {
        Self {
            inner,
            remaining: n,
        }
    }
}

impl<E> Source<E> for TakeSource<E> {
    fn next(&mut self) -> Result<Option<E>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        match self.inner.next()? {
            Some(e) => {
                self.remaining -= 1;
                Ok(Some(e))
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.inner.size_hint();
        let low = low.min(self.remaining);
        let high = high.map(|h| h.min(self.remaining));
        (low, high)
    }
}

pub struct EnumerateSource<E> {
    inner: Box<dyn Source<E>>,
    idx: usize,
}

impl<E> EnumerateSource<E> {
    pub fn new(inner: Box<dyn Source<E>>) -> Self {
        Self { inner, idx: 0 }
    }
}

impl<E> Source<(usize, E)> for EnumerateSource<E>
where
    E: Send,
{
    fn next(&mut self) -> Result<Option<(usize, E)>> {
        match self.inner.next()? {
            Some(e) => {
                let i = self.idx;
                self.idx += 1;
                Ok(Some((i, e)))
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
