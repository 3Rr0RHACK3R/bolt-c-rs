#![deny(unused_must_use)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngStreamState {
    pub key: u64,
    pub counter: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct RngStream {
    key: u64,
    counter: u64,
}

impl RngStream {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            key: mix64(seed),
            counter: 0,
        }
    }

    pub fn state(&self) -> RngStreamState {
        RngStreamState {
            key: self.key,
            counter: self.counter,
        }
    }

    pub fn set_state(&mut self, state: RngStreamState) {
        self.key = state.key;
        self.counter = state.counter;
    }

    pub fn fold_in(&self, value: u64) -> Self {
        Self {
            key: mix64(self.key ^ mix64(value)),
            counter: 0,
        }
    }

    pub fn split(&mut self) -> Self {
        let child_key = self.next_u64();
        Self {
            key: child_key,
            counter: 0,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let x = self.key.wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        mix64(x)
    }

    pub fn next_f64_01(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) / ((1u64 << 53) as f64)
    }

    pub fn next_f32_01(&mut self) -> f32 {
        let u = self.next_u64() >> 40;
        (u as f32) / ((1u32 << 24) as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngStreamsState {
    pub dropout: RngStreamState,
    pub data: RngStreamState,
    pub noise: RngStreamState,
}

#[derive(Clone, Copy, Debug)]
pub struct RngStreams {
    pub dropout: RngStream,
    pub data: RngStream,
    pub noise: RngStream,
}

impl RngStreams {
    pub fn from_seed(seed: u64) -> Self {
        let mut base = RngStream::from_seed(seed);
        Self {
            dropout: base.split(),
            data: base.split(),
            noise: base.split(),
        }
    }

    pub fn state(&self) -> RngStreamsState {
        RngStreamsState {
            dropout: self.dropout.state(),
            data: self.data.state(),
            noise: self.noise.state(),
        }
    }

    pub fn set_state(&mut self, state: RngStreamsState) {
        self.dropout.set_state(state.dropout);
        self.data.set_state(state.data);
        self.noise.set_state(state.noise);
    }

    pub fn fold_in(&self, value: u64) -> Self {
        Self {
            dropout: self.dropout.fold_in(value),
            data: self.data.fold_in(value),
            noise: self.noise.fold_in(value),
        }
    }

    pub fn split(&mut self) -> Self {
        Self {
            dropout: self.dropout.split(),
            data: self.data.split(),
            noise: self.noise.split(),
        }
    }

    pub fn split2(&mut self) -> (Self, Self) {
        (
            self.split(),
            self.split(),
        )
    }
}

pub fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
