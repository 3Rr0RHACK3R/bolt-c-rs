use bolt_core::backend::{Backend, RandomOp, TensorParts};
use bolt_core::error::Result;
use bolt_core::layout::Layout;
use bolt_core::shape::ConcreteShape;
use bolt_core::dtype::Float;
use bolt_core::allocator::StorageAllocator;
use crate::backend::CpuBackend;
use rand::{Rng, SeedableRng};
use rand::distributions::Uniform;
use rand_distr::{StandardNormal, Distribution};

impl<D: Float> RandomOp<D> for CpuBackend {
    fn uniform(
        &self,
        shape: &[usize],
        low: D,
        high: D,
        seed: Option<u64>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };
        
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let l: f64 = D::to_f64(low);
        let h: f64 = D::to_f64(high);
        let dist = Uniform::new(l, h);

        for i in 0..numel {
            let val: f64 = dist.sample(&mut rng);
            slice[i] = D::from_f64(val);
        }

        let layout = Layout::contiguous(shape);
        Ok(TensorParts { storage, layout })
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: D,
        std: D,
        seed: Option<u64>,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let mut storage = self.allocator().allocate(numel)?;
        let slice = unsafe { storage.try_as_mut_slice()? };

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let m: f64 = D::to_f64(mean);
        let s: f64 = D::to_f64(std);
        
        for i in 0..numel {
            let val: f64 = rng.sample(StandardNormal);
            let scaled = val * s + m;
            slice[i] = D::from_f64(scaled);
        }

        let layout = Layout::contiguous(shape);
        Ok(TensorParts { storage, layout })
    }
}
