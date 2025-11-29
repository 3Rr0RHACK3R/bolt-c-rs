use std::collections::HashMap;

use bolt_core::backend::AddOp;
use bolt_core::{Backend, Tensor};

use crate::error::Result;
use crate::{Float, GradTensor, Handle};

pub struct Gradients<B, D>
where
    B: Backend<D>,
    D: Float,
{
    grads: HashMap<Handle, Tensor<B, D>>,
    generation: u32,
}

impl<B, D> Gradients<B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub(crate) fn new(grads: HashMap<Handle, Tensor<B, D>>, generation: u32) -> Self {
        Self { grads, generation }
    }

    pub fn wrt(&self, tensor: &GradTensor<'_, B, D>) -> Option<&Tensor<B, D>> {
        if tensor.handle().generation != self.generation {
            return None;
        }
        self.grads.get(&tensor.handle())
    }

    pub fn get(&self, handle: &Handle) -> Option<&Tensor<B, D>> {
        if handle.generation != self.generation {
            return None;
        }
        self.grads.get(handle)
    }

    pub fn contains(&self, tensor: &GradTensor<'_, B, D>) -> bool {
        self.wrt(tensor).is_some()
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Handle, &Tensor<B, D>)> {
        self.grads.iter()
    }

    pub fn take(&mut self, tensor: &GradTensor<'_, B, D>) -> Option<Tensor<B, D>> {
        if tensor.handle().generation != self.generation {
            return None;
        }
        self.grads.remove(&tensor.handle())
    }

    pub fn accumulate(&mut self, other: &Gradients<B, D>) -> Result<()>
    where
        B: AddOp<D>,
    {
        for (handle, grad) in &other.grads {
            insert_or_accumulate(&mut self.grads, *handle, grad.clone())?;
        }
        Ok(())
    }
}

pub(crate) fn insert_or_accumulate<B, D>(
    grads: &mut HashMap<Handle, Tensor<B, D>>,
    handle: Handle,
    grad: Tensor<B, D>,
) -> Result<()>
where
    B: Backend<D> + AddOp<D>,
    D: Float,
{
    match grads.entry(handle) {
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let existing = entry.get();
            let new_grad = existing.add(&grad)?;
            entry.insert(new_grad);
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(grad);
        }
    }
    Ok(())
}
