//! Stable parameter identifiers.

use std::sync::atomic::{AtomicU64, Ordering};

/// Stable identifier for parameters and buffers.
/// Assigned once on creation; never changes within a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(u64);

impl ParamId {
    /// Create from raw u64 (for deserialization).
    #[inline]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get raw u64 value (for serialization).
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ParamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "param:{}", self.0)
    }
}

/// Thread-safe generator for unique `ParamId` values.
pub struct ParamIdGen {
    counter: AtomicU64,
}

impl ParamIdGen {
    /// Create a new generator starting from ID 1.
    pub const fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
        }
    }

    /// Generate the next unique `ParamId`.
    pub fn next(&self) -> ParamId {
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        ParamId(id)
    }
}

impl Default for ParamIdGen {
    fn default() -> Self {
        Self::new()
    }
}
