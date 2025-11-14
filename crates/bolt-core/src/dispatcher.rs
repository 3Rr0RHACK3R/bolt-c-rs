use std::sync::OnceLock;
use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::{
    error::{Error, Result},
    op::{OpAttrs, OpKey},
    tensor::Tensor,
};

pub type KernelFn = dyn Fn(&[Tensor], &OpAttrs) -> Result<Vec<Tensor>> + Send + Sync + 'static;

#[derive(Clone, Copy, Debug)]
pub enum KernelLayoutReq {
    Contiguous,
    GeneralStrided,
}

#[derive(Default)]
pub struct Dispatcher {
    table: HashMap<OpKey, KernelEntry>,
}

struct KernelEntry {
    layout_req: KernelLayoutReq,
    func: Arc<KernelFn>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        key: OpKey,
        layout_req: KernelLayoutReq,
        kernel: Arc<KernelFn>,
    ) -> Result<()> {
        if self.table.contains_key(&key) {
            return Err(Error::KernelAlreadyRegistered {
                op: key.op,
                device: key.device,
                dtype: key.dtype,
            });
        }
        self.table.insert(
            key,
            KernelEntry {
                layout_req,
                func: kernel,
            },
        );
        Ok(())
    }

    pub fn dispatch(&self, key: OpKey, inputs: &[Tensor], attrs: &OpAttrs) -> Result<Vec<Tensor>> {
        let entry = self.table.get(&key).ok_or(Error::KernelNotFound {
            op: key.op,
            device: key.device,
            dtype: key.dtype,
        })?;
        let prepared_inputs: Cow<'_, [Tensor]> = match entry.layout_req {
            KernelLayoutReq::GeneralStrided => Cow::Borrowed(inputs),
            KernelLayoutReq::Contiguous => {
                let converted = inputs
                    .iter()
                    .map(|tensor| {
                        if tensor.is_contiguous() {
                            Ok(tensor.clone())
                        } else {
                            tensor.contiguous()
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Cow::Owned(converted)
            }
        };
        (entry.func)(prepared_inputs.as_ref(), attrs)
    }
}

static DISPATCHER: OnceLock<Dispatcher> = OnceLock::new();

pub fn init_dispatcher<F>(build: F) -> Result<()>
where
    F: FnOnce(&mut Dispatcher) -> Result<()>,
{
    if DISPATCHER.get().is_some() {
        return Err(Error::DispatcherInitialized);
    }
    let mut dispatcher = Dispatcher::new();
    build(&mut dispatcher)?;
    DISPATCHER
        .set(dispatcher)
        .map_err(|_| Error::DispatcherInitialized)
}

pub fn global_dispatcher() -> Result<&'static Dispatcher> {
    DISPATCHER.get().ok_or(Error::DispatcherUninitialized)
}
