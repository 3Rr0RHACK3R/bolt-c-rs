use bolt_rng::{ModelRng, ModelRngState};

use crate::{CheckpointWriter, Result, SaveCheckpoint};

impl SaveCheckpoint for ModelRng {
    fn save(&self, w: &mut CheckpointWriter) -> Result<()> {
        let state = self.state();

        w.u64("rng.init.key", state.init.key)?;
        w.u64("rng.init.counter", state.init.counter)?;
        w.u64("rng.forward.key", state.forward.key)?;
        w.u64("rng.forward.counter", state.forward.counter)?;
        w.u64("rng.data.key", state.data.key)?;
        w.u64("rng.data.counter", state.data.counter)?;

        Ok(())
    }
}

impl crate::LoadCheckpoint for ModelRng {
    fn load(&mut self, r: &mut crate::CheckpointReader) -> Result<()> {
        let state = ModelRngState {
            init: bolt_rng::RngStreamState {
                key: r.u64("rng.init.key")?,
                counter: r.u64("rng.init.counter")?,
            },
            forward: bolt_rng::RngStreamState {
                key: r.u64("rng.forward.key")?,
                counter: r.u64("rng.forward.counter")?,
            },
            data: bolt_rng::RngStreamState {
                key: r.u64("rng.data.key")?,
                counter: r.u64("rng.data.counter")?,
            },
        };

        self.set_state(state);
        Ok(())
    }
}
