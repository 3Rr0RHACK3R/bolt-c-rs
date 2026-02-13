# Bolt

A deep learning framework written in Rust. If you come from PyTorch this will feel familiar, but entirely in Rust, with minimal dependencies.

---

## What is this?

Bolt is a modern tensor library and neural network framework built entirely in Rust. It gives you the flexibility to build and train models with the performance benefits of Rust's zero-cost abstractions.

I've been working on this because I wanted a DL framework that:

- Doesn't drag Python along for the ride
- Actually catches bugs at compile time
- Ships to environments where Python isn't welcome

It's still early, but the core pieces are in place: tensors, autograd, layers, optimizers, and checkpointing.

---

## Getting started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bolt-tensor = "0.1"
bolt-cpu = "0.1"
bolt-nn = "0.1"
bolt-optim = "0.1"
```

### Basic Tensor Operations

```rust
use std::sync::Arc;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_slice(&backend, &[4.0f32, 3.0, 2.0, 1.0], &[2, 2])?;

    let sum = a.add(&b)?;
    println!("Sum: {:?}", sum.to_vec()?); // [5.0, 5.0, 5.0, 5.0]

    Ok(())
}
```

### Training a Model

Here's a complete example of training a simple linear regression model ($y = 2x + 1$).

```rust
use std::sync::Arc;
use bolt_cpu::CpuBackend;
use bolt_nn::{ForwardCtx, Module, Store, layers::Linear};
use bolt_optim::{Sgd, SgdCfg};
use bolt_rng::RngKey;
use bolt_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = Arc::new(CpuBackend::new());

    let root_key = RngKey::from_seed(1337);
    let store = Store::new_with_init_key(backend.clone(), root_key.derive("init"));

    let layer = Linear::init(&store.sub("linear"), 1, 1, true)?;
    store.seal();

    let params = store.trainable();
    let mut opt = Sgd::new(SgdCfg {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    })?;

    let x = Tensor::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4, 1])?;
    let y = Tensor::from_slice(&backend, &[3.0, 5.0, 7.0, 9.0], &[4, 1])?;

    for step in 0..100 {
        store.zero_grad();

        let step_key = root_key.derive("forward").fold_in(step as u64);
        let mut ctx = ForwardCtx::train_with_key(step_key);
        let output = layer.forward(x.clone(), &mut ctx)?;

        let diff = output.sub(&y)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        store.backward(&loss)?;
        opt.step(&params)?;

        if step % 20 == 0 {
            println!("step {}: loss={:.4}", step, loss.item()?);
        }
    }

    Ok(())
}
```

### Inference

Running the model is just a forward pass with an evaluation context.

```rust
// ... (assuming model is trained)

let mut ctx = ForwardCtx::eval();
let test_x = Tensor::from_slice(&backend, &[5.0], &[1, 1])?;
let pred = layer.forward(test_x, &mut ctx)?;

println!("Prediction for x=5: {:.2}", pred.item()?); // Should be ~11.0
```

### Running Examples

You can run the included examples to see more complex models (like MNIST):

```bash
cargo run -p bolt-examples
```

---

## What works now

- **Tensors**: creation, slicing, reshaping, broadcasting
- **Autograd**: backward pass with gradients
- **CPU backend**: optimized operations, SIMD-friendly
- **Layers**: Linear, Conv2d, BatchNorm, Dropout, Pooling, etc.
- **Optimizers**: SGD (momentum supported)
- **Loss functions**: CrossEntropy, MSELoss, L1Loss
- **Checkpointing**: save/load with sharding, streaming, and checksum verification
- **Data utilities**: streams, batches, MNIST dataset

## What I'm working on

GPU support is next. CUDA first, then Metal for Apple Silicon. After that: more optimizers (Adam, RMSprop), ONNX import/export, and the usual grab-bag of quality-of-life improvements.

---

## Crates

Bolt is a workspace. Use what you need:

| crate            | what it does                               |
| ---------------- | ------------------------------------------ |
| `bolt-core`      | tensor types, layouts, storage, allocators |
| `bolt-tensor`    | tensor operations + autograd               |
| `bolt-cpu`       | CPU backend                                |
| `bolt-nn`        | layers and module system                   |
| `bolt-optim`     | optimizers                                 |
| `bolt-losses`    | loss functions                             |
| `bolt-serialize` | checkpoint I/O                             |
| `bolt-data`      | data loading streams                       |
| `bolt-datasets`  | built-in datasets                          |
| `bolt-vision`    | vision utilities                           |
| `bolt-rng`       | random number generation                   |
| `bolt-profiler`  | performance profiling                      |
| `bolt-benchmark` | operation benchmarks                       |

---

## Who this is for

- Rust developers who need ML without the Python dependency
- Embedded/edge scenarios where a Python runtime is a non-starter
- Anyone curious about building a DL framework in Rust

---

## Contributing

Contributions are welcome. The project has specific conventions around code style, testing, and architecture. It's just to keep the codebase readable and maintainable.

---
