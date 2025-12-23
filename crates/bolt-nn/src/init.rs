use std::f64::consts::PI;

use bolt_core::Float;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub enum Init<D: Float> {
    Zeros,
    Uniform { low: D, high: D },
    Normal { mean: D, std: D },
    KaimingUniform { a: D },
    XavierUniform,
    XavierNormal,
}

#[derive(Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f64(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as u64;
        (u as f64) / ((1u64 << 53) as f64)
    }

    pub fn uniform_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    pub fn normal_f64(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z0
    }
}

pub fn fill<D: Float>(shape: &[usize], init: Init<D>, rng: &mut Rng) -> Result<Vec<D>> {
    let numel: usize = shape.iter().product();
    let mut out = vec![D::zero(); numel];

    match init {
        Init::Zeros => {}
        Init::Uniform { low, high } => {
            let low = low.to_f64();
            let high = high.to_f64();
            for x in &mut out {
                *x = D::from_f64(rng.uniform_f64(low, high));
            }
        }
        Init::Normal { mean, std } => {
            let mean = mean.to_f64();
            let std = std.to_f64();
            for x in &mut out {
                *x = D::from_f64(rng.normal_f64(mean, std));
            }
        }
        Init::KaimingUniform { a } => {
            if shape.len() < 2 {
                return Err(Error::State(
                    "kaiming_uniform expects at least 2D weight".into(),
                ));
            }
            let fan_in = shape[1] as f64;
            let a = a.to_f64();
            let gain = (2.0 / (1.0 + a * a)).sqrt();
            let std = gain / fan_in.sqrt();
            let bound = (3.0f64).sqrt() * std;
            for x in &mut out {
                *x = D::from_f64(rng.uniform_f64(-bound, bound));
            }
        }
        Init::XavierUniform => {
            if shape.len() < 2 {
                return Err(Error::State(
                    "xavier_uniform expects at least 2D weight".into(),
                ));
            }
            let fan_in = shape[1] as f64;
            let fan_out = shape[0] as f64;
            let bound = (6.0 / (fan_in + fan_out)).sqrt();
            for x in &mut out {
                *x = D::from_f64(rng.uniform_f64(-bound, bound));
            }
        }
        Init::XavierNormal => {
            if shape.len() < 2 {
                return Err(Error::State(
                    "xavier_normal expects at least 2D weight".into(),
                ));
            }
            let fan_in = shape[1] as f64;
            let fan_out = shape[0] as f64;
            let std = (2.0 / (fan_in + fan_out)).sqrt();
            for x in &mut out {
                *x = D::from_f64(rng.normal_f64(0.0, std));
            }
        }
    }

    Ok(out)
}
