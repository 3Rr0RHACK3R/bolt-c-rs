use bolt_core::Error as CoreError;

#[derive(Debug)]
pub enum Error {
    Core(CoreError),
    Shape(String),
    State(String),
}

impl From<CoreError> for Error {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Core(e) => write!(f, "{e}"),
            Error::Shape(s) => write!(f, "shape error: {s}"),
            Error::State(s) => write!(f, "state error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
