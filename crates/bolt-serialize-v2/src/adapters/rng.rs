use bolt_rng::RngKey;

use crate::{CheckpointReader, CheckpointWriter, Result, LoadCheckpoint, SaveCheckpoint};

impl SaveCheckpoint for RngKey {
    fn save(&self, w: &mut CheckpointWriter) -> Result<()> {
        w.u64("key", self.key())?;
        Ok(())
    }
}

impl LoadCheckpoint for RngKey {
    fn load(&mut self, r: &mut CheckpointReader) -> Result<()> {
        let key_value = r.u64("key")?;
        *self = RngKey::from_seed(key_value);
        Ok(())
    }
}
