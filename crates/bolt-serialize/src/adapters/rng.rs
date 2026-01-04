use bolt_core::{DType, shape::Shape};
use bolt_rng::ModelRng;

use crate::{Checkpoint, Record, RecordMeta, Result, Role};

pub trait RngCheckpointAdapter {
    fn to_records(&self) -> Box<dyn Iterator<Item = Result<Record<'static>>> + '_>;
    fn restore_from_checkpoint(&mut self, ckpt: &Checkpoint) -> Result<()>;
}

impl RngCheckpointAdapter for ModelRng {
    fn to_records(&self) -> Box<dyn Iterator<Item = Result<Record<'static>>> + '_> {
        let state = self.state();

        let records = vec![
            ("rng.init.key", state.init.key),
            ("rng.init.counter", state.init.counter),
            ("rng.forward.key", state.forward.key),
            ("rng.forward.counter", state.forward.counter),
            ("rng.data.key", state.data.key),
            ("rng.data.counter", state.data.counter),
        ];

        Box::new(records.into_iter().map(|(name, value)| {
            let shape = Shape::from_slice(&[8]).expect("shape [8] is valid");
            let meta = RecordMeta::new(name, DType::U8, shape).with_role(Role::Rng);
            let data = value.to_le_bytes().to_vec();
            Ok(Record::new(meta, data))
        }))
    }

    fn restore_from_checkpoint(&mut self, ckpt: &Checkpoint) -> Result<()> {
        fn read_u64(ckpt: &Checkpoint, name: &str) -> Result<u64> {
            let view = ckpt.get(name)?;
            if view.dtype != DType::U8 {
                return Err(crate::Error::RestoreFailed {
                    reason: format!("expected dtype U8 for '{name}', got {:?}", view.dtype),
                });
            }
            if view.data.len() != 8 {
                return Err(crate::Error::RestoreFailed {
                    reason: format!("expected 8 bytes for '{name}', got {}", view.data.len()),
                });
            }
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(view.data);
            Ok(u64::from_le_bytes(bytes))
        }

        let init_key = read_u64(ckpt, "rng.init.key")?;
        let init_counter = read_u64(ckpt, "rng.init.counter")?;
        let forward_key = read_u64(ckpt, "rng.forward.key")?;
        let forward_counter = read_u64(ckpt, "rng.forward.counter")?;
        let data_key = read_u64(ckpt, "rng.data.key")?;
        let data_counter = read_u64(ckpt, "rng.data.counter")?;

        let state = bolt_rng::ModelRngState {
            init: bolt_rng::RngStreamState {
                key: init_key,
                counter: init_counter,
            },
            forward: bolt_rng::RngStreamState {
                key: forward_key,
                counter: forward_counter,
            },
            data: bolt_rng::RngStreamState {
                key: data_key,
                counter: data_counter,
            },
        };

        self.set_state(state);
        Ok(())
    }
}
