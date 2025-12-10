use bolt_autodiff::ParamId;
use bolt_core::Error as CoreError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("learning rate must be positive, got {value}")]
    InvalidLearningRate { value: f64 },

    #[error("momentum must be non-negative, got {value}")]
    InvalidMomentum { value: f64 },

    #[error("weight decay must be non-negative, got {value}")]
    InvalidWeightDecay { value: f64 },

    #[error("parameter not registered in optimizer: id={param_id:?}, name={param_name:?}")]
    UnknownParameter {
        param_id: ParamId,
        param_name: Option<String>,
    },

    #[error("missing gradient for parameter: id={param_id:?}, name={param_name:?}")]
    MissingGradient {
        param_id: ParamId,
        param_name: Option<String>,
    },

    #[error(
        "gradient shape mismatch for parameter id={param_id:?}, name={param_name:?}: grad={grad_shape:?}, param={param_shape:?}"
    )]
    ShapeMismatch {
        param_id: ParamId,
        param_name: Option<String>,
        grad_shape: Vec<usize>,
        param_shape: Vec<usize>,
    },

    #[error(transparent)]
    Core(#[from] CoreError),
}
