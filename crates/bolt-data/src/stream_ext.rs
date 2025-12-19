use crate::{MapWithSource, Result, Stream, TryMapSource, TryMapWithSource};

impl<E> Stream<E> {
    pub fn try_map<F, E2>(self, f: F) -> Stream<E2>
    where
        F: Fn(E) -> Result<E2> + Send + Sync + 'static,
        E: Send + 'static,
        E2: Send + 'static,
    {
        Stream::new(TryMapSource {
            inner: self.inner,
            f,
        })
    }

    pub fn map_with<C, F, E2>(self, ctx: C, f: F) -> Stream<E2>
    where
        C: Send + 'static,
        F: Fn(&C, E) -> E2 + Send + Sync + 'static,
        E: Send + 'static,
        E2: Send + 'static,
    {
        Stream::new(MapWithSource {
            inner: self.inner,
            ctx,
            f,
        })
    }

    pub fn try_map_with<C, F, E2>(self, ctx: C, f: F) -> Stream<E2>
    where
        C: Send + 'static,
        F: Fn(&C, E) -> Result<E2> + Send + Sync + 'static,
        E: Send + 'static,
        E2: Send + 'static,
    {
        Stream::new(TryMapWithSource {
            inner: self.inner,
            ctx,
            f,
        })
    }
}
