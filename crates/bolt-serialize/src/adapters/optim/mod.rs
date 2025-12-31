mod sgd;

use bolt_core::{BaseBackend, Float};
use bolt_nn::Store;

use crate::{Checkpoint, Record, Result};

pub trait OptimizerCheckpointAdapter<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn to_records<'a>(
        &'a self,
        store: &'a Store<B, D>,
    ) -> Box<dyn Iterator<Item = Result<Record<'static>>> + 'a>;

    fn restore_from_checkpoint(&mut self, ckpt: &Checkpoint, store: &Store<B, D>) -> Result<()>;
}
