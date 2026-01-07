use bolt_rng::RngKey;

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct ForwardCtx {
    train: bool,
    rng_key: Option<RngKey>,
    stochastic_counter: u64,
}

impl ForwardCtx {
    fn new(train: bool, rng_key: Option<RngKey>) -> Self {
        Self {
            train,
            rng_key,
            stochastic_counter: 0,
        }
    }

    /// Create a training context with an RNG key.
    ///
    /// The key is domainless - layers derive their own subkeys using `derive_stochastic_key`.
    pub fn train_with_key(key: RngKey) -> Self {
        Self::new(true, Some(key))
    }

    /// Create an evaluation context (no randomness needed).
    pub fn eval() -> Self {
        Self::new(false, None)
    }

    pub fn is_train(&self) -> bool {
        self.train
    }

    /// Derive a key for a stochastic operation.
    ///
    /// This automatically handles collision avoidance by:
    /// 1. Deriving a domain-specific key (e.g., "dropout", "noise")
    /// 2. Folding in a counter that increments per call
    ///
    /// This ensures:
    /// - Different domains are independent
    /// - Same domain, different calls get different keys
    /// - Deterministic for same call order
    pub fn derive_stochastic_key(&mut self, domain: &str) -> Option<RngKey> {
        let base_key = self.rng_key?;
        let counter = self.stochastic_counter;
        self.stochastic_counter += 1;

        // Derive: base_key.derive(domain).fold_in(counter)
        // This ensures independence and determinism
        Some(base_key.derive(domain).fold_in(counter))
    }

    /// Derive a key with an explicit path (for advanced use cases).
    ///
    /// This doesn't increment the counter, so it's useful for deterministic
    /// behavior when you want to control the exact derivation path.
    pub fn derive_with_path(&self, path: &[&str]) -> Option<RngKey> {
        self.rng_key.map(|k| k.derive_path(path))
    }

    /// Split the context into two independent contexts.
    ///
    /// Useful for parallel branches that need independent randomness.
    pub fn split(&mut self) -> Result<Self> {
        let Some(key) = self.rng_key else {
            return Err(Error::State(
                "ForwardCtx::split requires RNG key; use ForwardCtx::train_with_key".into(),
            ));
        };
        let (k1, k2) = key.split();
        self.rng_key = Some(k1);
        Ok(Self::train_with_key(k2))
    }

    /// Split the context into three independent contexts.
    ///
    /// Returns (self_updated, child1, child2).
    pub fn split2(&mut self) -> Result<(Self, Self)> {
        let Some(key) = self.rng_key else {
            return Err(Error::State(
                "ForwardCtx::split2 requires RNG key; use ForwardCtx::train_with_key".into(),
            ));
        };
        let keys = key.split_n(3);
        self.rng_key = Some(keys[0]);
        Ok((Self::train_with_key(keys[1]), Self::train_with_key(keys[2])))
    }
}
