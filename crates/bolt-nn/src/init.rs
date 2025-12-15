use bolt_core::backend::{FillOp, RandomOp};
use bolt_core::dtype::Float;
use bolt_core::error::Result;
use bolt_core::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Nonlinearity {
    ReLU,
    LeakyReLU,
    Tanh,
    Sigmoid,
    Linear,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FanMode {
    FanIn,
    FanOut,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Init {
    KaimingUniform {
        a: f32,
        mode: FanMode,
        nonlinearity: Nonlinearity,
    },
    XavierUniform {
        gain: f32,
    },
    Normal {
        mean: f32,
        std: f32,
    },
    Uniform {
        low: f32,
        high: f32,
    },
    Constant {
        val: f32,
    },
    Zeros,
    Ones,
}

impl Init {
    pub fn init<B, D>(&self, backend: &Arc<B>, shape: &[usize]) -> Result<Tensor<B, D>>
    where
        B: RandomOp<D> + FillOp<D>,
        D: Float,
    {
        match self {
            Init::KaimingUniform {
                a,
                mode,
                nonlinearity,
            } => {
                let fan = calculate_fan(shape, *mode);
                let gain = calculate_gain(*nonlinearity, *a);
                let std = gain / (fan as f32).sqrt();
                let bound = (3.0f32).sqrt() * std;
                let low = D::from_f64(-bound as f64);
                let high = D::from_f64(bound as f64);
                Tensor::uniform(backend, shape, low, high, None)
            }
            Init::XavierUniform { gain } => {
                let (fan_in, fan_out) = calculate_fans(shape);
                let std = *gain * (2.0 / (fan_in as f32 + fan_out as f32)).sqrt();
                let bound = (3.0f32).sqrt() * std;
                let low = D::from_f64(-bound as f64);
                let high = D::from_f64(bound as f64);
                Tensor::uniform(backend, shape, low, high, None)
            }
            Init::Normal { mean, std } => {
                let m = D::from_f64(*mean as f64);
                let s = D::from_f64(*std as f64);
                Tensor::normal(backend, shape, m, s, None)
            }
            Init::Uniform { low, high } => {
                let l = D::from_f64(*low as f64);
                let h = D::from_f64(*high as f64);
                Tensor::uniform(backend, shape, l, h, None)
            }
            Init::Constant { val } => {
                let v = D::from_f64(*val as f64);
                Tensor::full(backend, shape, v)
            }
            Init::Zeros => Tensor::full(backend, shape, D::zero()),
            Init::Ones => Tensor::full(backend, shape, D::one()),
        }
    }
}

fn calculate_fan(shape: &[usize], mode: FanMode) -> usize {
    let (fan_in, fan_out) = calculate_fans(shape);
    match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    }
}

fn calculate_fans(shape: &[usize]) -> (usize, usize) {
    if shape.len() < 2 {
        let size = if shape.is_empty() { 1 } else { shape[0] };
        return (size, size);
    }
    let num_input_fmaps = shape[1];
    let num_output_fmaps = shape[0];
    let mut receptive_field_size = 1;
    if shape.len() > 2 {
        receptive_field_size = shape[2..].iter().product();
    }
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;
    (fan_in, fan_out)
}

fn calculate_gain(nonlinearity: Nonlinearity, param: f32) -> f32 {
    match nonlinearity {
        Nonlinearity::Linear => 1.0,
        Nonlinearity::ReLU => (2.0f32).sqrt(),
        Nonlinearity::LeakyReLU => (2.0f32 / (1.0 + param.powi(2))).sqrt(),
        Nonlinearity::Tanh => 5.0 / 3.0,
        Nonlinearity::Sigmoid => 1.0,
    }
}
