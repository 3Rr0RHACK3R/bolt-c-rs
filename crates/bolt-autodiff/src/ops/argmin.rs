use bolt_core::Error;

pub fn argmin_not_differentiable() -> Error {
    Error::OpError("argmin is not differentiable".into())
}
