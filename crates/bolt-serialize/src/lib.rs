mod error;
mod format;
mod load;
mod options;
mod record;
mod save;

pub mod adapters;

pub use error::{Error, Result};
pub use format::SHARDS_DIR;
pub use load::{Checkpoint, CheckpointInfo, CheckpointMeta, inspect, load_checkpoint};
pub use options::{LoadOpts, OnError, RestoreOpts, RestoreReport, SaveOpts};
pub use record::{Record, RecordMeta, RecordView, Role};
pub use save::{ShardEntry, ShardPlan, plan_shards, save_checkpoint};

pub use adapters::optim::OptimizerCheckpointAdapter;
pub use adapters::rng::RngCheckpointAdapter;
pub use adapters::store::StoreCheckpointAdapter;
pub use adapters::tensor::{TensorFromCheckpoint, TensorToRecord};
