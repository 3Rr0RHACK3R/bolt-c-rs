use bolt_rng::RngStreams;

use crate::{Error, Result};

#[derive(Debug)]
pub struct ForwardCtx {
    train: bool,
    rngs: Option<RngStreams>,
}

impl ForwardCtx {
    fn new(train: bool, rngs: Option<RngStreams>) -> Self {
        Self { train, rngs }
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
