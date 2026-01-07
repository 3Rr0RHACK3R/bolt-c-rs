#![deny(unused_must_use)]

/// A stateless, copyable RNG key for deterministic random number generation.
///
/// `RngKey` follows JAX-style design: keys are immutable and can be split/derived
/// to create independent streams. All randomness comes from converting a key to
/// `RngSeq` via `into_seq()`.
///
/// # Example
/// ```
/// use bolt_rng::RngKey;
///
/// let root = RngKey::from_seed(42);
/// let init_key = root.derive("init");
/// let param_key = init_key.derive_path(&["params", "layer1", "w"]);
/// let mut seq = param_key.into_seq();
/// let value = seq.next_f32_01();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RngKey {
    key: u64,
}

impl RngKey {
    /// Create a new RNG key from a seed.
    pub fn from_seed(seed: u64) -> Self {
        Self { key: mix64(seed) }
    }

    /// Create a new RNG key from entropy (nondeterministic).
    ///
    /// This is explicitly nondeterministic and should only be used when
    /// reproducibility is not required.
    pub fn from_entropy() -> Self {
        Self::from_seed(entropy_seed())
    }

    /// Derive a new key by mixing in a string tag.
    ///
    /// This creates a deterministic child key based on the tag. Different tags
    /// produce independent streams.
    pub fn derive(self, tag: &str) -> Self {
        let tag_hash = fnv1a_64(tag.as_bytes());
        Self {
            key: mix64(self.key ^ mix64(tag_hash)),
        }
    }

    /// Derive a new key from a hierarchical path of tags.
    ///
    /// This is equivalent to calling `derive` multiple times, but more efficient.
    /// Useful for module/layer paths like `["dropout", "Block1/Dropout"]`.
    pub fn derive_path(self, tags: &[&str]) -> Self {
        let mut key = self;
        for tag in tags {
            key = key.derive(tag);
        }
        key
    }

    /// Fold in a numeric value to create a new key.
    ///
    /// Useful for incorporating step numbers, epoch numbers, device IDs, etc.
    pub fn fold_in(self, value: u64) -> Self {
        Self {
            key: mix64(self.key ^ mix64(value)),
        }
    }

    /// Split this key into two independent keys.
    pub fn split(self) -> (Self, Self) {
        let mut seq = self.into_seq();
        let k1 = Self::from_seed(seq.next_u64());
        let k2 = Self::from_seed(seq.next_u64());
        (k1, k2)
    }

    /// Split this key into `n` independent keys.
    ///
    /// Returns a vector of `n` keys that are all independent from each other
    /// and from the original key.
    pub fn split_n(self, n: usize) -> Vec<Self> {
        let mut seq = self.into_seq();
        (0..n)
            .map(|_| Self::from_seed(seq.next_u64()))
            .collect()
    }

    /// Convert this key into a sequence generator for sampling.
    ///
    /// The sequence is deterministic: the same key always produces the same sequence.
    pub fn into_seq(self) -> RngSeq {
        RngSeq {
            key: self.key,
            counter: 0,
        }
    }

    /// Get the internal key value (for serialization/debugging).
    pub fn key(&self) -> u64 {
        self.key
    }
}

/// A sequence generator for sampling random values from an RngKey.
///
/// `RngSeq` is created from `RngKey::into_seq()` and provides methods for
/// generating random numbers. It maintains an internal counter that advances
/// with each sample.
#[derive(Clone, Copy, Debug)]
pub struct RngSeq {
    key: u64,
    counter: u64,
}

impl RngSeq {
    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let x = self.key.wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        mix64(x)
    }

    /// Generate a random f64 in [0, 1).
    pub fn next_f64_01(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) / ((1u64 << 53) as f64)
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32_01(&mut self) -> f32 {
        let u = self.next_u64() >> 40;
        (u as f32) / ((1u32 << 24) as f32)
    }

    /// Generate a random usize in the given range [start, end).
    ///
    /// Uses rejection sampling to ensure uniform distribution.
    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let start = range.start;
        let end = range.end;
        if start >= end {
            return start;
        }
        let len = end - start;
        if len == 1 {
            return start;
        }

        let len_u64 = len as u64;
        let threshold = u64::MAX - (u64::MAX % len_u64);

        loop {
            let value = self.next_u64();
            if value < threshold {
                return start + ((value % len_u64) as usize);
            }
        }
    }
}

/// Generates an entropy-based seed using system time and an atomic counter.
///
/// This is explicitly nondeterministic and should only be used when
/// reproducibility is not required.
pub fn entropy_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENTROPY_COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let ctr = ENTROPY_COUNTER.fetch_add(1, Ordering::Relaxed);
    mix64(nanos ^ ctr)
}

/// Mix a 64-bit value using splitmix-like hashing.
pub fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// FNV-1a 64-bit hash function.
///
/// A simple, stable hash function that produces consistent results across
/// Rust versions. Used for deterministic key derivation from string tags.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
