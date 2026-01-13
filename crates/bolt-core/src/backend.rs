use crate::{
    allocator::StorageAllocator,
    device::{BackendDevice, DeviceId, DeviceKind},
    dtype::{Float, NativeType},
    error::Result,
    layout::Layout,
    shape::Shape,
};

#[derive(Clone, Debug)]
pub struct TensorParts<S> {
    pub storage: S,
    pub layout: Layout,
}

pub trait Backend: Clone + Send + Sync + 'static {
    type Device: BackendDevice + Clone + Send + Sync + 'static;
    type Storage<D: NativeType>: Clone + Send + Sync + 'static;
    type Allocator<D: NativeType>: StorageAllocator<D, Storage = Self::Storage<D>>;

    fn device(&self) -> &Self::Device;
    fn device_id(&self) -> DeviceId {
        self.device().device_id()
    }
    fn allocator<D: NativeType>(&self) -> Self::Allocator<D>;
    fn device_kind(&self) -> DeviceKind {
        self.device().kind()
    }
    fn storage_len_bytes<D: NativeType>(&self, storage: &Self::Storage<D>) -> usize;

    fn read<D: NativeType>(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        dst: &mut [D],
    ) -> Result<()>;
    fn write<D: NativeType>(
        &self,
        storage: &mut Self::Storage<D>,
        layout: &Layout,
        src: &[D],
    ) -> Result<()>;
}

pub trait CopyOp<D: NativeType>: Backend {
    fn copy(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait CastOp<Src: NativeType, Dst: NativeType>: Backend {
    fn cast(
        &self,
        storage: &Self::Storage<Src>,
        layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<Dst>>>;
}

pub trait FillOp<D: NativeType>: Backend {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage<D>>;
}

pub trait AddOp<D: NativeType>: Backend {
    fn add(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait MulOp<D: NativeType>: Backend {
    fn mul(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SubOp<D: NativeType>: Backend {
    fn sub(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait MatmulOp<D: NativeType>: Backend {
    fn matmul(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait MeanOp<D: Float>: Backend {
    fn mean(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait NegOp<D: NativeType>: Backend {
    fn neg(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait AbsOp<D: NativeType>: Backend {
    fn abs(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait ExpOp<D: Float>: Backend {
    fn exp(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait LogOp<D: Float>: Backend {
    fn log(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SqrtOp<D: Float>: Backend {
    fn sqrt(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SinOp<D: Float>: Backend {
    fn sin(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait CosOp<D: Float>: Backend {
    fn cos(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait TanhOp<D: Float>: Backend {
    fn tanh(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait ReluOp<D: NativeType>: Backend {
    fn relu(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SigmoidOp<D: Float>: Backend {
    fn sigmoid(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait DivOp<D: NativeType>: Backend {
    fn div(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait PowOp<D: Float>: Backend {
    fn pow(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SumOp<D: NativeType>: Backend {
    fn sum(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait ProdOp<D: NativeType>: Backend {
    fn prod(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait MinOp<D: NativeType>: Backend {
    fn min(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait MaxOp<D: NativeType>: Backend {
    fn max(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait ArgminOp<D: NativeType>: Backend {
    fn argmin(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<i32>>>;
}

pub trait ArgmaxOp<D: NativeType>: Backend {
    fn argmax(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<TensorParts<Self::Storage<i32>>>;
}

pub trait ReshapeOp<D: NativeType>: Backend {
    fn reshape(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        new_shape: &[usize],
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let shape = Shape::from_slice(new_shape)?;
        let new_layout = layout.reshape(shape)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}

pub trait SqueezeOp<D: NativeType>: Backend {
    fn squeeze_all(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let new_layout = layout.squeeze_all()?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }

    fn squeeze_axis(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        axis: isize,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let new_layout = layout.squeeze_axis(axis)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}

pub trait UnsqueezeOp<D: NativeType>: Backend {
    fn unsqueeze_axis(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        axis: isize,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let new_layout = layout.unsqueeze_axis(axis)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}

pub trait TransposeOp<D: NativeType>: Backend {
    fn transpose(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        axis_a: isize,
        axis_b: isize,
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let new_layout = layout.transpose(axis_a, axis_b)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}

pub trait BroadcastToOp<D: NativeType>: Backend {
    fn broadcast_to(
        &self,
        storage: &Self::Storage<D>,
        layout: &Layout,
        shape: &[usize],
    ) -> Result<TensorParts<Self::Storage<D>>> {
        let target_shape = Shape::from_slice(shape)?;
        let new_layout = layout.broadcast_to(&target_shape)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}

pub trait RandomOp<D: NativeType>: Backend {
    fn uniform(
        &self,
        shape: &[usize],
        low: D,
        high: D,
        key: bolt_rng::RngKey,
    ) -> Result<TensorParts<Self::Storage<D>>>;

    fn normal(
        &self,
        shape: &[usize],
        mean: D,
        std: D,
        key: bolt_rng::RngKey,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait BernoulliMaskOp<D: Float>: Backend {
    fn bernoulli_mask(
        &self,
        shape: &[usize],
        p_keep: D,
        key: bolt_rng::RngKey,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait ConcatOp<D: NativeType>: Backend {
    fn concat(
        &self,
        tensors: &[(&Self::Storage<D>, &Layout)],
        axis: usize,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}
