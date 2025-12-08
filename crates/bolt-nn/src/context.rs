use std::cell::RefCell;
use std::sync::Arc;

use bolt_core::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Train,
    Eval,
}

pub struct Context<B: Backend> {
    backend: Arc<B>,
    mode: RefCell<Mode>,
    rng: RefCell<Rng>,
}

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl<B: Backend> Context<B> {
    pub fn train(backend: Arc<B>, seed: u64) -> Self {
        Self {
            backend,
            mode: RefCell::new(Mode::Train),
            rng: RefCell::new(Rng::new(seed)),
        }
    }

    pub fn infer(backend: Arc<B>) -> Self {
        Self {
            backend,
            mode: RefCell::new(Mode::Eval),
            rng: RefCell::new(Rng::new(0)),
        }
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    pub fn mode(&self) -> Mode {
        *self.mode.borrow()
    }

    pub fn is_training(&self) -> bool {
        self.mode() == Mode::Train
    }

    pub fn set_train(&self) {
        *self.mode.borrow_mut() = Mode::Train;
    }

    pub fn set_eval(&self) {
        *self.mode.borrow_mut() = Mode::Eval;
    }

    pub fn rng(&self) -> std::cell::RefMut<'_, Rng> {
        self.rng.borrow_mut()
    }

    pub fn evaluating<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let prev = self.mode();
        self.set_eval();
        let result = f();
        *self.mode.borrow_mut() = prev;
        result
    }
}
