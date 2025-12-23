use std::f64::consts::PI;

use bolt_core::Float;
use bolt_rng::RngStream;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub enum Init<D: Float> {
    Zeros,
    Ones,
    Uniform { low: D, high: D },
    Normal { mean: D, std: D },
    KaimingUniform { a: D },
    KaimingNormal { a: D },
    XavierUniform,
    XavierNormal,
}

fn uniform_f64(rng: &mut RngStream, low: f64, high: f64) -> f64 {
    low + (high - low) * rng.next_f64_01()
}

fn normal_f64(rng: &mut RngStream, mean: f64, std: f64) -> f64 {
    let u1 = rng.next_f64_01().max(1e-12);
    let u2 = rng.next_f64_01();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std * z0
}

pub fn fill<D: Float>(shape: &[usize], init: Init<D>, rng: &mut RngStream) -> Result<Vec<D>> {
    let numel: usize = shape.iter().product();
    let mut out = vec![D::zero(); numel];

    match init {
        Init::Zeros => {}
        Init::Ones => {
            for x in &mut out {
                *x = D::one();
            }
        }
        Init::Uniform { low, high } => {
            let low = low.to_f64();
            let high = high.to_f64();
            for x in &mut out {
                *x = D::from_f64(uniform_f64(rng, low, high));
            }
        }
        Init::Normal { mean, std } => {
            let mean = mean.to_f64();
            let std = std.to_f64();
            for x in &mut out {
                *x = D::from_f64(normal_f64(rng, mean, std));
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
                *x = D::from_f64(uniform_f64(rng, -bound, bound));
            }
        }
        Init::KaimingNormal { a } => {
            if shape.len() < 2 {
                return Err(Error::State(
                    "kaiming_normal expects at least 2D weight".into(),
                ));
            }
            let fan_in = shape[1] as f64;
            let a = a.to_f64();
            let gain = (2.0 / (1.0 + a * a)).sqrt();
            let std = gain / fan_in.sqrt();
            for x in &mut out {
                *x = D::from_f64(normal_f64(rng, 0.0, std));
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
                *x = D::from_f64(uniform_f64(rng, -bound, bound));
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
                *x = D::from_f64(normal_f64(rng, 0.0, std));
            }
        }
    }

    Ok(out)
}
