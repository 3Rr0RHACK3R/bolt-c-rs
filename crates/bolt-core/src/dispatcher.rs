use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::{
    device::DeviceKind,
    dtype::DType,
    error::{Error, Result},
    op::{OpAttrs, OpKey, Operation},
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

    pub fn register_operation<O, F>(
        &mut self,
        device: DeviceKind,
        dtype: DType,
        layout_req: KernelLayoutReq,
        kernel: F,
    ) -> Result<()>
    where
        O: Operation,
        F: Fn(&[Tensor], &O) -> Result<Vec<Tensor>> + Send + Sync + 'static,
    {
        let key = OpKey {
            op: O::KIND,
            device,
            dtype,
        };
        let func: Arc<KernelFn> = Arc::new(move |inputs, attrs| {
            let op = O::from_opattrs(attrs)?;
            kernel(inputs, &op)
        });
        self.register(key, layout_req, func)
    }
}
