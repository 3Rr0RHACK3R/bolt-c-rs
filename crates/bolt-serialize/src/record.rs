use bolt_core::{dtype::DType, shape::Shape};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Metadata for a single record in a checkpoint.
#[derive(Clone, Debug)]
pub struct RecordMeta {
    pub key: String,
    pub dtype: DType,
    pub shape: Shape,
    pub offset: u64,
    pub length: u64,
    pub shard_id: usize,
}

impl Serialize for RecordMeta {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RecordMeta", 6)?;
        state.serialize_field("key", &self.key)?;
        state.serialize_field("dtype", &self.dtype)?;
        state.serialize_field("shape", self.shape.as_slice())?;
        state.serialize_field("offset", &self.offset)?;
        state.serialize_field("length", &self.length)?;
        state.serialize_field("shard_id", &self.shard_id)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RecordMeta {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct RecordMetaVisitor;

        impl<'de> Visitor<'de> for RecordMetaVisitor {
            type Value = RecordMeta;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct RecordMeta")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<RecordMeta, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut key = None;
                let mut dtype = None;
                let mut shape_vec: Option<Vec<usize>> = None;
                let mut offset = None;
                let mut length = None;
                let mut shard_id = None;

                while let Some(k) = map.next_key()? {
                    match k {
                        "key" => {
                            if key.is_some() {
                                return Err(de::Error::duplicate_field("key"));
                            }
                            key = Some(map.next_value()?);
                        }
                        "dtype" => {
                            if dtype.is_some() {
                                return Err(de::Error::duplicate_field("dtype"));
                            }
                            dtype = Some(map.next_value()?);
                        }
                        "shape" => {
                            if shape_vec.is_some() {
                                return Err(de::Error::duplicate_field("shape"));
                            }
                            shape_vec = Some(map.next_value()?);
                        }
                        "offset" => {
                            if offset.is_some() {
                                return Err(de::Error::duplicate_field("offset"));
                            }
                            offset = Some(map.next_value()?);
                        }
                        "length" => {
                            if length.is_some() {
                                return Err(de::Error::duplicate_field("length"));
                            }
                            length = Some(map.next_value()?);
                        }
                        "shard_id" => {
                            if shard_id.is_some() {
                                return Err(de::Error::duplicate_field("shard_id"));
                            }
                            shard_id = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }

                let key = key.ok_or_else(|| de::Error::missing_field("key"))?;
                let dtype = dtype.ok_or_else(|| de::Error::missing_field("dtype"))?;
                let shape_vec = shape_vec.ok_or_else(|| de::Error::missing_field("shape"))?;
                let shape = Shape::from_slice(&shape_vec)
                    .map_err(|e| de::Error::custom(format!("Invalid shape: {}", e)))?;
                let offset = offset.ok_or_else(|| de::Error::missing_field("offset"))?;
                let length = length.ok_or_else(|| de::Error::missing_field("length"))?;
                let shard_id = shard_id.ok_or_else(|| de::Error::missing_field("shard_id"))?;

                Ok(RecordMeta {
                    key,
                    dtype,
                    shape,
                    offset,
                    length,
                    shard_id,
                })
            }
        }

        deserializer.deserialize_map(RecordMetaVisitor)
    }
}

/// A record to be written to a checkpoint.
pub struct Record {
    pub key: String,
    pub dtype: DType,
    pub shape: Shape,
    pub data: Vec<u8>,
}

/// A view into a record (for lazy loading).
pub struct RecordView {
    pub meta: RecordMeta,
    pub data: Vec<u8>,
}

impl Record {
    pub fn new(key: String, dtype: DType, shape: Shape, data: Vec<u8>) -> Self {
        Self {
            key,
            dtype,
            shape,
            data,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

impl RecordView {
    #[allow(dead_code)]
    pub fn new(meta: RecordMeta, data: Vec<u8>) -> Self {
        Self { meta, data }
    }
}
