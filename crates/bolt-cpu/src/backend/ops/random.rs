use crate::backend::CpuBackend;
use bolt_core::allocator::StorageAllocator;
use bolt_core::backend::{Backend, BernoulliMaskOp, RandomOp, TensorParts};
use bolt_core::dtype::Float;
use bolt_core::error::{Error, Result};
use bolt_core::layout::Layout;
use bolt_core::shape::Shape;
use bolt_rng::RngKey;

impl<D: Float> RandomOp<D> for CpuBackend {
    fn uniform(
        &self,
        shape: &[usize],
        low: D,
        high: D,
        key: RngKey,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let numel: usize = shape.iter().product();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };
        let l: f64 = D::to_f64(low);
        let h: f64 = D::to_f64(high);

        let mut seq = key.into_seq();
        let span = h - l;
        for elem in slice.iter_mut() {
            let u = seq.next_f64_01();
            *elem = D::from_f64(l + span * u);
        }

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: D,
        std: D,
        key: RngKey,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let numel: usize = shape.iter().product();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };
        let m: f64 = D::to_f64(mean);
        let s: f64 = D::to_f64(std);

        let mut seq = key.into_seq();
        let mut i = 0;
        while i < numel {
            let mut u1 = seq.next_f64_01();
            if u1 == 0.0 {
                u1 = f64::MIN_POSITIVE;
            }
            let u2 = seq.next_f64_01();
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

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }
}

impl<D: Float> BernoulliMaskOp<D> for CpuBackend {
    fn bernoulli_mask(
        &self,
        shape: &[usize],
        p_keep: D,
        key: RngKey,
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

        let mut seq = key.into_seq();
        for out in slice.iter_mut() {
            let u = seq.next_f64_01();
            *out = if u < p { D::one() } else { D::zero() };
        }

        let layout = Layout::contiguous(Shape::from_slice(shape)?);
        Ok(TensorParts { storage, layout })
    }
}
