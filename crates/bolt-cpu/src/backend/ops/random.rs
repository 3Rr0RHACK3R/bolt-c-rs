use crate::backend::CpuBackend;
use bolt_core::allocator::StorageAllocator;
use bolt_core::backend::{Backend, BernoulliMaskOp, RandomOp, TensorParts};
use bolt_core::dtype::Float;
use bolt_core::error::{Error, Result};
use bolt_core::layout::Layout;
use bolt_core::shape::Shape;
use bolt_rng::RngStream;
use rand::Rng;
use rand::SeedableRng;

impl<D: Float> RandomOp<D> for CpuBackend {
    fn uniform(
        &self,
        shape: &[usize],
        low: D,
        high: D,
        seed: Option<u64>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let numel: usize = shape.iter().product();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };
        let l: f64 = D::to_f64(low);
        let h: f64 = D::to_f64(high);

        match seed {
            Some(s) => {
                let mut stream = RngStream::from_seed(s);
                let span = h - l;
                for elem in slice.iter_mut() {
                    let u = stream.next_f64_01();
                    *elem = D::from_f64(l + span * u);
                }
            }
            None => {
                let mut rng = rand::rngs::StdRng::from_entropy();
                let entropy_seed: u64 = rng.r#gen();
                let mut stream = RngStream::from_seed(entropy_seed);
                let span = h - l;
                for elem in slice.iter_mut() {
                    let u = stream.next_f64_01();
                    *elem = D::from_f64(l + span * u);
                }
            }
        }

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: D,
        std: D,
        seed: Option<u64>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let numel: usize = shape.iter().product();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };
        let m: f64 = D::to_f64(mean);
        let s: f64 = D::to_f64(std);

        match seed {
            Some(seed) => {
                let mut stream = RngStream::from_seed(seed);
                let mut i = 0;
                while i < numel {
                    let mut u1 = stream.next_f64_01();
                    if u1 == 0.0 {
                        u1 = f64::MIN_POSITIVE;
                    }
                    let u2 = stream.next_f64_01();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    let z0 = r * theta.cos();
                    let z1 = r * theta.sin();

                    slice[i] = D::from_f64(z0 * s + m);
                    if i + 1 < numel {
                        slice[i + 1] = D::from_f64(z1 * s + m);
                    }
                    i += 2;
                }
            }
            None => {
                let mut rng = rand::rngs::StdRng::from_entropy();
                let entropy_seed: u64 = rng.r#gen();
                let mut stream = RngStream::from_seed(entropy_seed);
                let mut i = 0;
                while i < numel {
                    let mut u1 = stream.next_f64_01();
                    if u1 == 0.0 {
                        u1 = f64::MIN_POSITIVE;
                    }
                    let u2 = stream.next_f64_01();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    let z0 = r * theta.cos();
                    let z1 = r * theta.sin();

                    slice[i] = D::from_f64(z0 * s + m);
                    if i + 1 < numel {
                        slice[i + 1] = D::from_f64(z1 * s + m);
                    }
                    i += 2;
                }
            }
        }

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }
}

impl<D: Float> BernoulliMaskOp<D> for CpuBackend {
    fn bernoulli_mask(
        &self,
        shape: &[usize],
        p_keep: D,
        seed: Option<u64>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let p = D::to_f64(p_keep);
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::OpError(format!(
                "bernoulli_mask: p_keep must be in [0, 1], got {p}"
            )));
        }

        let numel: usize = shape.iter().product();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };

        match seed {
            Some(seed) => {
                let mut stream = RngStream::from_seed(seed);
                for out in slice.iter_mut() {
                    let u = stream.next_f64_01();
                    *out = if u < p { D::one() } else { D::zero() };
                }
            }
            None => {
                let mut rng = rand::rngs::StdRng::from_entropy();
                let entropy_seed: u64 = rng.r#gen();
                let mut stream = RngStream::from_seed(entropy_seed);
                for out in slice.iter_mut() {
                    let u = stream.next_f64_01();
                    *out = if u < p { D::one() } else { D::zero() };
                }
            }
        }

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }
}
