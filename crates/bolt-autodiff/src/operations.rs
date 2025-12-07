use std::marker::PhantomData;
use std::sync::Arc;

use bolt_core::Tensor;
use bolt_core::backend::{
    AbsOp, AddOp, ArgmaxOp, ArgminOp, Backend, BroadcastToOp, CopyOp, CosOp, DivOp, ExpOp, FillOp,
    LogOp, MatmulOp, MaxOp, MeanOp, MinOp, MulOp, NegOp, PowOp, ProdOp, ReluOp, ReshapeOp, SinOp,
    SqrtOp, SqueezeOp, SubOp, SumOp, TanhOp, TensorParts, TransposeOp, UnsqueezeOp,
};
use bolt_core::layout::Layout;

use crate::backward::{BackwardOp, MAX_INPUTS};
use crate::graph::Graph;
use crate::ops::{
    AbsBackward, AddBackward, CosBackward, DivBackward, ExpBackward, ExpandBackward, LogBackward,
    MatmulBackward, MeanBackward, MulBackward, NegBackward, PowBackward, ReluBackward,
    ReshapeBackward, SinBackward, SqrtBackward, SqueezeBackward, SubBackward, SumBackward,
    TanhBackward, TransposeBackward, UnsqueezeBackward,
};
use crate::storage::AutodiffStorage;
use crate::{Float, Handle};

pub struct Autodiff<B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub(crate) inner: Arc<B>,
    pub(crate) graph: Arc<std::sync::RwLock<Option<Graph<B, D>>>>,
    pub(crate) grad_enabled: Arc<std::sync::RwLock<bool>>,
    pub(crate) _marker: PhantomData<D>,
}

impl<B, D> Autodiff<B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub fn is_grad_enabled(&self) -> bool {
        *self.grad_enabled.read().unwrap()
    }

    pub fn has_active_graph(&self) -> bool {
        self.graph.read().unwrap().is_some()
    }

    pub(crate) fn with_graph<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Graph<B, D>) -> R,
    {
        let mut graph_ref = self.graph.write().unwrap();
        graph_ref.as_mut().map(f)
    }

    pub(crate) fn create_tracked_storage(
        &self,
        inner_storage: B::Storage,
        _layout: &Layout,
        requires_grad: bool,
        is_leaf: bool,
        inputs: tinyvec::ArrayVec<[Handle; MAX_INPUTS]>,
        backward_op: Option<Box<dyn BackwardOp<B, D>>>,
        saved_tensors: Vec<bolt_core::Tensor<B, D>>,
    ) -> AutodiffStorage<B::Storage> {
        let handle = self
            .with_graph(|graph| {
                graph.create_node(requires_grad, is_leaf, inputs, backward_op, saved_tensors)
            })
            .unwrap_or(Handle::NONE);

        AutodiffStorage::new(inner_storage, handle, requires_grad)
    }

    // Helper for binary operations
    fn binary_requires_grad(
        &self,
        lhs: &AutodiffStorage<B::Storage>,
        rhs: &AutodiffStorage<B::Storage>,
    ) -> bool {
        (lhs.requires_grad || rhs.requires_grad)
            && self.is_grad_enabled()
            && self.has_active_graph()
    }

    // Helper for unary operations
    fn unary_requires_grad(&self, storage: &AutodiffStorage<B::Storage>) -> bool {
        storage.requires_grad && self.is_grad_enabled() && self.has_active_graph()
    }

    // Helper for early return (no gradient needed)
    fn no_grad_result(parts: TensorParts<B::Storage>) -> TensorParts<AutodiffStorage<B::Storage>> {
        TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        }
    }

    // Helper for binary operations with tracked storage
    fn create_tracked_binary_op(
        &self,
        parts: TensorParts<B::Storage>,
        lhs_handle: Handle,
        rhs_handle: Handle,
        backward_op: Box<dyn BackwardOp<B, D>>,
        saved_tensors: Vec<Tensor<B, D>>,
    ) -> TensorParts<AutodiffStorage<B::Storage>> {
        let mut inputs = tinyvec::ArrayVec::new();
        inputs.push(lhs_handle);
        inputs.push(rhs_handle);

        let storage = self.create_tracked_storage(
            parts.storage,
            &parts.layout,
            true,
            false,
            inputs,
            Some(backward_op),
            saved_tensors,
        );

        TensorParts {
            storage,
            layout: parts.layout,
        }
    }

    // Helper for unary operations with tracked storage
    fn create_tracked_unary_op(
        &self,
        parts: TensorParts<B::Storage>,
        input_handle: Handle,
        backward_op: Box<dyn BackwardOp<B, D>>,
        saved_tensors: Vec<Tensor<B, D>>,
    ) -> TensorParts<AutodiffStorage<B::Storage>> {
        let mut inputs = tinyvec::ArrayVec::new();
        inputs.push(input_handle);

        let storage = self.create_tracked_storage(
            parts.storage,
            &parts.layout,
            true,
            false,
            inputs,
            Some(backward_op),
            saved_tensors,
        );

        TensorParts {
            storage,
            layout: parts.layout,
        }
    }
}

impl<B, D> CopyOp<D> for Autodiff<B, D>
where
    B: Backend<D> + CopyOp<D>,
    D: Float,
{
    fn copy(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.copy(&storage.inner, layout)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, storage.handle, storage.requires_grad),
            layout: parts.layout,
        })
    }
}

impl<B, D> FillOp<D> for Autodiff<B, D>
where
    B: Backend<D> + FillOp<D>,
    D: Float,
{
    fn fill(&self, layout: &Layout, value: D) -> bolt_core::Result<Self::Storage> {
        let inner = self.inner.fill(layout, value)?;
        Ok(AutodiffStorage::new(inner, Handle::NONE, false))
    }
}

impl<B, D> AddOp<D> for Autodiff<B, D>
where
    B: Backend<D> + AddOp<D> + SumOp<D> + CopyOp<D>,
    D: Float,
{
    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .add(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op =
            AddBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![],
        ))
    }
}

impl<B, D> SubOp<D> for Autodiff<B, D>
where
    B: Backend<D> + SubOp<D> + AddOp<D> + FillOp<D> + SumOp<D> + CopyOp<D>,
    D: Float,
{
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .sub(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op =
            SubBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![],
        ))
    }
}

impl<B, D> MulOp<D> for Autodiff<B, D>
where
    B: Backend<D> + MulOp<D> + AddOp<D> + SumOp<D> + CopyOp<D>,
    D: Float,
{
    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .mul(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let lhs_tensor =
            Tensor::from_parts(self.inner.clone(), lhs.inner.clone(), lhs_layout.clone());
        let rhs_tensor =
            Tensor::from_parts(self.inner.clone(), rhs.inner.clone(), rhs_layout.clone());
        let backward_op =
            MulBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![lhs_tensor, rhs_tensor],
        ))
    }
}

impl<B, D> MatmulOp<D> for Autodiff<B, D>
where
    B: Backend<D> + MatmulOp<D> + AddOp<D> + SumOp<D> + CopyOp<D> + TransposeOp<D>,
    D: Float,
{
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .matmul(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let lhs_tensor =
            Tensor::from_parts(self.inner.clone(), lhs.inner.clone(), lhs_layout.clone());
        let rhs_tensor =
            Tensor::from_parts(self.inner.clone(), rhs.inner.clone(), rhs_layout.clone());

        let backward_op =
            MatmulBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![lhs_tensor, rhs_tensor],
        ))
    }
}

impl<B, D> SumOp<D> for Autodiff<B, D>
where
    B: Backend<D> + SumOp<D> + AddOp<D> + FillOp<D> + CopyOp<D> + ReshapeOp<D> + BroadcastToOp<D>,
    D: Float,
{
    fn sum(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.sum(layout, &storage.inner, axes, keepdims)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_shape = layout.shape().to_vec();
        let normalized_axes = axes
            .map(|a| bolt_core::shape::canonical_axes(a, input_shape.len()))
            .transpose()?;
        let backward_op = SumBackward::new(input_shape, normalized_axes);

        let mut inputs = tinyvec::ArrayVec::new();
        inputs.push(storage.handle);

        let out_storage = self.create_tracked_storage(
            parts.storage,
            &parts.layout,
            true,
            false,
            inputs,
            Some(Box::new(backward_op)),
            vec![],
        );

        Ok(TensorParts {
            storage: out_storage,
            layout: parts.layout,
        })
    }
}

impl<B, D> MeanOp<D> for Autodiff<B, D>
where
    B: Backend<D>
        + MeanOp<D>
        + AddOp<D>
        + FillOp<D>
        + MulOp<D>
        + CopyOp<D>
        + ReshapeOp<D>
        + BroadcastToOp<D>,
    D: Float,
{
    fn mean(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.mean(layout, &storage.inner, axes, keepdims)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_shape = layout.shape().to_vec();
        let count = match axes {
            None => layout.num_elements(),
            Some(ax) => {
                let canonical = bolt_core::shape::canonical_axes(ax, input_shape.len())?;
                canonical.iter().map(|&a| input_shape[a]).product()
            }
        };

        let normalized_axes = axes
            .map(|a| bolt_core::shape::canonical_axes(a, input_shape.len()))
            .transpose()?;
        let backward_op = MeanBackward::new(input_shape, normalized_axes, count);

        let mut inputs = tinyvec::ArrayVec::new();
        inputs.push(storage.handle);

        let out_storage = self.create_tracked_storage(
            parts.storage,
            &parts.layout,
            true,
            false,
            inputs,
            Some(Box::new(backward_op)),
            vec![],
        );

        Ok(TensorParts {
            storage: out_storage,
            layout: parts.layout,
        })
    }
}

impl<B, D> NegOp<D> for Autodiff<B, D>
where
    B: Backend<D> + NegOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn neg(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.neg(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op = NegBackward::new();

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}

impl<B, D> AbsOp<D> for Autodiff<B, D>
where
    B: Backend<D> + AbsOp<D> + AddOp<D> + DivOp<D> + MulOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn abs(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.abs(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_tensor =
            Tensor::from_parts(self.inner.clone(), storage.inner.clone(), layout.clone());

        let backward_op = AbsBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![input_tensor],
        ))
    }
}

impl<B, D> ExpOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ExpOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    fn exp(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.exp(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let output_tensor = Tensor::from_parts(
            self.inner.clone(),
            parts.storage.clone(),
            parts.layout.clone(),
        );

        let backward_op = ExpBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![output_tensor],
        ))
    }
}

impl<B, D> LogOp<D> for Autodiff<B, D>
where
    B: Backend<D> + LogOp<D> + DivOp<D> + CopyOp<D>,
    D: Float,
{
    fn log(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.log(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_tensor =
            Tensor::from_parts(self.inner.clone(), storage.inner.clone(), layout.clone());

        let backward_op = LogBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![input_tensor],
        ))
    }
}

impl<B, D> SqrtOp<D> for Autodiff<B, D>
where
    B: Backend<D> + SqrtOp<D> + DivOp<D> + MulOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn sqrt(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.sqrt(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let output_tensor = Tensor::from_parts(
            self.inner.clone(),
            parts.storage.clone(),
            parts.layout.clone(),
        );

        let backward_op = SqrtBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![output_tensor],
        ))
    }
}

impl<B, D> SinOp<D> for Autodiff<B, D>
where
    B: Backend<D> + SinOp<D> + MulOp<D> + CosOp<D> + CopyOp<D>,
    D: Float,
{
    fn sin(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.sin(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_tensor =
            Tensor::from_parts(self.inner.clone(), storage.inner.clone(), layout.clone());

        let backward_op = SinBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![input_tensor],
        ))
    }
}

impl<B, D> CosOp<D> for Autodiff<B, D>
where
    B: Backend<D> + CosOp<D> + SinOp<D> + MulOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn cos(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.cos(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_tensor =
            Tensor::from_parts(self.inner.clone(), storage.inner.clone(), layout.clone());

        let backward_op = CosBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![input_tensor],
        ))
    }
}

impl<B, D> TanhOp<D> for Autodiff<B, D>
where
    B: Backend<D> + TanhOp<D> + MulOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn tanh(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.tanh(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let output_tensor = Tensor::from_parts(
            self.inner.clone(),
            parts.storage.clone(),
            parts.layout.clone(),
        );

        let backward_op = TanhBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![output_tensor],
        ))
    }
}

impl<B, D> ReluOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ReluOp<D> + AddOp<D> + MulOp<D> + DivOp<D> + AbsOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn relu(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.relu(layout, &storage.inner)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let input_tensor =
            Tensor::from_parts(self.inner.clone(), storage.inner.clone(), layout.clone());

        let backward_op = ReluBackward::new();

        Ok(self.create_tracked_unary_op(
            parts,
            storage.handle,
            Box::new(backward_op),
            vec![input_tensor],
        ))
    }
}

impl<B, D> DivOp<D> for Autodiff<B, D>
where
    B: Backend<D> + DivOp<D> + AddOp<D> + MulOp<D> + SubOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .div(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let lhs_tensor =
            Tensor::from_parts(self.inner.clone(), lhs.inner.clone(), lhs_layout.clone());
        let rhs_tensor =
            Tensor::from_parts(self.inner.clone(), rhs.inner.clone(), rhs_layout.clone());

        let backward_op =
            DivBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![lhs_tensor, rhs_tensor],
        ))
    }
}

impl<B, D> PowOp<D> for Autodiff<B, D>
where
    B: Backend<D>
        + PowOp<D>
        + AddOp<D>
        + MulOp<D>
        + LogOp<D>
        + SubOp<D>
        + FillOp<D>
        + DivOp<D>
        + CopyOp<D>
        + SumOp<D>,
    D: Float,
{
    fn pow(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .pow(&lhs.inner, &rhs.inner, lhs_layout, rhs_layout)?;

        if !self.binary_requires_grad(lhs, rhs) {
            return Ok(Self::no_grad_result(parts));
        }

        let lhs_tensor =
            Tensor::from_parts(self.inner.clone(), lhs.inner.clone(), lhs_layout.clone());
        let rhs_tensor =
            Tensor::from_parts(self.inner.clone(), rhs.inner.clone(), rhs_layout.clone());
        let output_tensor = Tensor::from_parts(
            self.inner.clone(),
            parts.storage.clone(),
            parts.layout.clone(),
        );

        let backward_op =
            PowBackward::new(lhs_layout.shape().to_vec(), rhs_layout.shape().to_vec());

        Ok(self.create_tracked_binary_op(
            parts,
            lhs.handle,
            rhs.handle,
            Box::new(backward_op),
            vec![lhs_tensor, rhs_tensor, output_tensor],
        ))
    }
}

impl<B, D> ProdOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ProdOp<D>,
    D: Float,
{
    fn prod(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.prod(layout, &storage.inner, axes, keepdims)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        })
    }
}

impl<B, D> MinOp<D> for Autodiff<B, D>
where
    B: Backend<D> + MinOp<D>,
    D: Float,
{
    fn min(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.min(layout, &storage.inner, axes, keepdims)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        })
    }
}

impl<B, D> MaxOp<D> for Autodiff<B, D>
where
    B: Backend<D> + MaxOp<D>,
    D: Float,
{
    fn max(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.max(layout, &storage.inner, axes, keepdims)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        })
    }
}

impl<B, D> ArgminOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ArgminOp<D> + Backend<i32>,
    D: Float,
{
    type I32Storage = AutodiffStorage<<B as ArgminOp<D>>::I32Storage>;

    fn argmin(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::I32Storage>> {
        let parts =
            ArgminOp::<D>::argmin(self.inner.as_ref(), layout, &storage.inner, axes, keepdims)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        })
    }
}

impl<B, D> ArgmaxOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ArgmaxOp<D> + Backend<i32>,
    D: Float,
{
    type I32Storage = AutodiffStorage<<B as ArgmaxOp<D>>::I32Storage>;

    fn argmax(
        &self,
        layout: &Layout,
        storage: &Self::Storage,
        axes: Option<&[isize]>,
        keepdims: bool,
    ) -> bolt_core::Result<TensorParts<Self::I32Storage>> {
        let parts =
            ArgmaxOp::<D>::argmax(self.inner.as_ref(), layout, &storage.inner, axes, keepdims)?;
        Ok(TensorParts {
            storage: AutodiffStorage::new(parts.storage, Handle::NONE, false),
            layout: parts.layout,
        })
    }
}

impl<B, D> ReshapeOp<D> for Autodiff<B, D>
where
    B: Backend<D> + ReshapeOp<D> + CopyOp<D>,
    D: Float,
{
    fn reshape(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        new_shape: &[usize],
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.reshape(&storage.inner, layout, new_shape)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op = ReshapeBackward::new(layout.shape().to_vec());

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}

impl<B, D> SqueezeOp<D> for Autodiff<B, D>
where
    B: Backend<D> + SqueezeOp<D> + ReshapeOp<D> + CopyOp<D>,
    D: Float,
{
    fn squeeze_all(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.squeeze_all(&storage.inner, layout)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op = SqueezeBackward::new(layout.shape().to_vec());

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }

    fn squeeze_axis(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        axis: isize,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.squeeze_axis(&storage.inner, layout, axis)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op = SqueezeBackward::new(layout.shape().to_vec());

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}

impl<B, D> UnsqueezeOp<D> for Autodiff<B, D>
where
    B: Backend<D> + UnsqueezeOp<D> + CopyOp<D> + SqueezeOp<D>,
    D: Float,
{
    fn unsqueeze_axis(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        axis: isize,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.unsqueeze_axis(&storage.inner, layout, axis)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let rank = layout.shape().len();
        let normalized_axis = if axis < 0 {
            ((rank as isize) + 1 + axis) as usize
        } else {
            axis as usize
        };

        let backward_op = UnsqueezeBackward::new(normalized_axis);

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}

impl<B, D> TransposeOp<D> for Autodiff<B, D>
where
    B: Backend<D> + TransposeOp<D> + CopyOp<D>,
    D: Float,
{
    fn transpose(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        axis_a: isize,
        axis_b: isize,
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self
            .inner
            .transpose(&storage.inner, layout, axis_a, axis_b)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let rank = layout.shape().len();
        let normalized_a = if axis_a < 0 {
            ((rank as isize) + axis_a) as usize
        } else {
            axis_a as usize
        };
        let normalized_b = if axis_b < 0 {
            ((rank as isize) + axis_b) as usize
        } else {
            axis_b as usize
        };

        let backward_op = TransposeBackward::new(normalized_a, normalized_b);

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}

impl<B, D> BroadcastToOp<D> for Autodiff<B, D>
where
    B: Backend<D> + BroadcastToOp<D> + ReshapeOp<D> + SumOp<D> + CopyOp<D>,
    D: Float,
{
    fn broadcast_to(
        &self,
        storage: &Self::Storage,
        layout: &Layout,
        shape: &[usize],
    ) -> bolt_core::Result<TensorParts<Self::Storage>> {
        let parts = self.inner.broadcast_to(&storage.inner, layout, shape)?;

        if !self.unary_requires_grad(storage) {
            return Ok(Self::no_grad_result(parts));
        }

        let backward_op =
            ExpandBackward::new(layout.shape().to_vec(), parts.layout.shape().to_vec());

        Ok(self.create_tracked_unary_op(parts, storage.handle, Box::new(backward_op), vec![]))
    }
}
