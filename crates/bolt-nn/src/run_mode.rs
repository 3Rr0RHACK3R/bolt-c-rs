#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RunMode {
    Train,
    Eval,
}

pub trait Trainable {
    fn set_run_mode(&mut self, _mode: RunMode) {}

    fn train(&mut self) {
        self.set_run_mode(RunMode::Train)
    }

    fn eval(&mut self) {
        self.set_run_mode(RunMode::Eval)
    }
}
