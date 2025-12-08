use crate::{
    allocator::StorageAllocator,
    device::{BackendDevice, DeviceKind},
    dtype::{FloatType, NativeType},
    error::Result,
    layout::Layout,
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

pub trait MeanOp<D: FloatType>: Backend {
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

pub trait ExpOp<D: FloatType>: Backend {
    fn exp(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait LogOp<D: FloatType>: Backend {
    fn log(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SqrtOp<D: FloatType>: Backend {
    fn sqrt(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait SinOp<D: FloatType>: Backend {
    fn sin(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait CosOp<D: FloatType>: Backend {
    fn cos(
        &self,
        layout: &Layout,
        storage: &Self::Storage<D>,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait TanhOp<D: FloatType>: Backend {
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

pub trait DivOp<D: NativeType>: Backend {
    fn div(
        &self,
        lhs: &Self::Storage<D>,
        rhs: &Self::Storage<D>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage<D>>>;
}

pub trait PowOp<D: FloatType>: Backend {
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
        use crate::shape::ConcreteShape;
        let shape = ConcreteShape::from_slice(new_shape)?;
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
        use crate::shape::ConcreteShape;
        let target_shape = ConcreteShape::from_slice(shape)?;
        let new_layout = layout.broadcast_to(&target_shape)?;
        Ok(TensorParts {
            storage: storage.clone(),
            layout: new_layout,
        })
    }
}
