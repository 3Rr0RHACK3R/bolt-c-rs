use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bolt_rng::{RngStreams, mix64};

use crate::{Error, Result};

#[derive(Debug)]
pub struct ForwardCtx {
    train: bool,
    rngs: Option<RngStreams>,
}

static CTX_ENTROPY_COUNTER: AtomicU64 = AtomicU64::new(0);

fn entropy_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let ctr = CTX_ENTROPY_COUNTER.fetch_add(1, Ordering::Relaxed);
    mix64(nanos ^ ctr)
}

impl ForwardCtx {
    fn new(train: bool, rngs: Option<RngStreams>) -> Self {
        Self { train, rngs }
    }

    pub fn train() -> Self {
        Self::new(true, Some(RngStreams::from_seed(entropy_seed())))
    }

    pub fn train_with_rngs(rngs: RngStreams) -> Self {
        Self::new(true, Some(rngs))
    }

    pub fn eval() -> Self {
        Self::new(false, None)
    }

    pub fn is_train(&self) -> bool {
        self.train
    }

    pub fn rngs_mut(&mut self) -> Option<&mut RngStreams> {
        self.rngs.as_mut()
    }

    pub fn split(&mut self) -> Result<Self> {
        let Some(rngs) = self.rngs.as_mut() else {
            return Err(Error::State(
                "ForwardCtx::split requires RNG streams; use ForwardCtx::train_with_rngs".into(),
            ));
        };
        Ok(Self::train_with_rngs(rngs.split()))
    }

    pub fn split2(&mut self) -> Result<(Self, Self)> {
        let Some(rngs) = self.rngs.as_mut() else {
            return Err(Error::State(
                "ForwardCtx::split2 requires RNG streams; use ForwardCtx::train_with_rngs".into(),
            ));
        };
        let (left, right) = rngs.split2();
        Ok((Self::train_with_rngs(left), Self::train_with_rngs(right)))
    }
}
