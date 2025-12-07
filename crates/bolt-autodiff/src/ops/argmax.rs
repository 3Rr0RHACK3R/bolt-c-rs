use bolt_core::Error;

pub fn argmax_not_differentiable() -> Error {
    Error::OpError("argmax is not differentiable".into())
}
