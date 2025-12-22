# bolt-losses

Tiny loss/metric helpers for Bolt backends. Provides MSE, cross-entropy
(probabilities or logits), and top-1 accuracy that work with any `Backend`
supporting the required ops.

```rust
use std::sync::Arc;
use bolt_tensor::Tensor;
use bolt_cpu::CpuBackend;
use bolt_losses::{cross_entropy_from_logits, accuracy_top1, Reduction};

let backend = Arc::new(CpuBackend::new());
let logits = Tensor::<_, f32>::from_slice(&backend, &[2.0, 0.5, 0.1], &[1, 3])?;
let target = Tensor::<_, f32>::from_slice(&backend, &[1.0, 0.0, 0.0], &[1, 3])?;

let loss = cross_entropy_from_logits(&logits, &target, Reduction::Mean)?.item()?;
let labels = Tensor::<_, i32>::from_slice(&backend, &[0], &[1])?;
let acc = accuracy_top1(&logits, &labels)?;
```

The crate is re-exported from `bolt-nn` behind the `losses` feature (enabled by
default).
