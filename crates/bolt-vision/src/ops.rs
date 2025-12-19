use bolt_core::backend::{BroadcastToOp, CopyOp, DivOp, FillOp, ReshapeOp, SubOp};
use bolt_core::{Backend, NativeType, Tensor};

pub mod tensor {
    use super::*;

    // Optional: u8 -> f32 cast if needed by datasets not emitting f32
    pub fn to_f32<B>(img: Tensor<B, u8>) -> bolt_core::Result<Tensor<B, f32>>
    where
        B: Backend + CopyOp<u8>,
    {
        let shape = img.shape().to_vec();
        let data = img.to_vec()?;
        let casted: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
        Tensor::from_vec(&img.backend(), casted, &shape)
    }

    pub fn scale<B>(v: f32, img: Tensor<B, f32>) -> bolt_core::Result<Tensor<B, f32>>
    where
        B: Backend + FillOp<f32> + bolt_core::backend::MulOp<f32>,
    {
        let c = Tensor::full_like(&img, v)?;
        img.mul(&c)
    }

    pub fn normalize<B>(
        mean: &[f32],
        std: &[f32],
        layout: crate::types::ImageLayout,
        img: Tensor<B, f32>,
    ) -> bolt_core::Result<Tensor<B, f32>>
    where
        B: Backend + SubOp<f32> + DivOp<f32> + ReshapeOp<f32> + BroadcastToOp<f32>,
    {
        let shape = img.shape();
        let (h, w, c, nhwc) = match shape.len() {
            3 => match layout {
                crate::types::ImageLayout::NHWC => (shape[0], shape[1], shape[2], true),
                crate::types::ImageLayout::NCHW => (shape[1], shape[2], shape[0], false),
            },
            _ => {
                return Err(bolt_core::Error::invalid_shape(
                    "normalize expects 3D image tensor",
                ))
            }
        };
        if mean.len() != c || std.len() != c {
            return Err(bolt_core::Error::invalid_shape("channel params mismatch"));
        }

        let backend = img.backend();
        let mean_t = Tensor::from_slice(&backend, mean, &[c])?;
        let std_t = Tensor::from_slice(&backend, std, &[c])?;

        let mean_shaped = if nhwc {
            mean_t.reshape(&[1, 1, c])?
        } else {
            mean_t.reshape(&[c, 1, 1])?
        };
        let std_shaped = if nhwc {
            std_t.reshape(&[1, 1, c])?
        } else {
            std_t.reshape(&[c, 1, 1])?
        };

        let target = if nhwc { &[h, w, c][..] } else { &[c, h, w][..] };
        let mean_b = mean_shaped.broadcast_to(target)?;
        let std_b = std_shaped.broadcast_to(target)?;

        let centered = img.sub(&mean_b)?;
        centered.div(&std_b)
    }

    pub fn hwc_to_chw<B, D>(img: Tensor<B, D>) -> bolt_core::Result<Tensor<B, D>>
    where
        B: Backend,
        D: NativeType,
    {
        if img.rank() != 3 {
            return Err(bolt_core::Error::invalid_shape("hwc_to_chw expects rank 3"));
        }
        img.permute(&[2, 0, 1])
    }

    pub fn resize_nn_u8<B>(
        img: Tensor<B, u8>,
        h: usize,
        w: usize,
        c: usize,
        new_h: usize,
        new_w: usize,
    ) -> bolt_core::Result<Tensor<B, u8>>
    where
        B: Backend + CopyOp<u8>,
    {
        let data = img.to_vec()?;
        let mut out = vec![0u8; new_h * new_w * c];
        for oy in 0..new_h {
            let iy = oy * h / new_h;
            for ox in 0..new_w {
                let ix = ox * w / new_w;
                for ch in 0..c {
                    let src = (iy * w + ix) * c + ch;
                    let dst = (oy * new_w + ox) * c + ch;
                    out[dst] = data[src];
                }
            }
        }
        Tensor::from_vec(&img.backend(), out, &[new_h, new_w, c])
    }
}
