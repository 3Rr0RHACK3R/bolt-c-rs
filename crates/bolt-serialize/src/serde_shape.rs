//! Serde adapter for `Shape` that serializes as a JSON array `[1, 2, 3]`
//! while validating shape constraints (rank <= 12, no zero dims, no overflow) on deserialize.

use bolt_core::shape::Shape;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

pub fn serialize<S>(shape: &Shape, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    shape.as_slice().serialize(serializer)
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Shape, D::Error>
where
    D: Deserializer<'de>,
{
    let dims: Vec<usize> = Vec::deserialize(deserializer)?;
    Shape::from_slice(&dims).map_err(de::Error::custom)
}
