# Bolt: A Modern Eager‑First Architecture (2025+)

> This document describes a new end‑to‑end design for Bolt as if we were starting fresh in 2025, with the benefit of hindsight from PyTorch, JAX, TensorFlow, Burn and others.
> It is intentionally verbose and conversational so that a junior engineer can read it and get a concrete mental model, not just a bunch of abstract bullets.

---

## 0. How to Read This Document

This is not a short RFC. It is a “living blueprint” for a modern deep learning framework built in Rust.

If you are new to the codebase or to DL systems, read it in this order:

1. Big picture: Sections 1–3.
2. Core concepts: Sections 4–7.
3. Execution flow, tracing, and compilation: Sections 8–11.
4. Memory, backends, and kernels: Sections 12–13.
5. Tooling and roadmap: Sections 14–16.

You do _not_ have to agree with every choice. The goal is to make the choices explicit and reasoned, so that disagreement is concrete instead of vague.

Throughout this doc we will include Rust‑like code sketches. They are not final code and will not compile as‑is; they exist to make the ideas concrete.

---

## 1. Motivation: Why Change at All?

Bolt today looks more like early TensorFlow 1.x than PyTorch 2.x:

- You build an explicit computation graph.
- You compile it to an internal program representation.
- You execute it using a VM.

This is a great starting point for a compiler paper. It is less great as a day‑to‑day tool for researchers and ML engineers in 2025.

Why?

Because the world has largely moved to _eager‑first_ frameworks:

- PyTorch: immediate execution, dynamic shapes, simple control flow.
- JAX: Pythonic API with JIT; graph/IR lives underneath, not as the main user surface.
- Burn: eager Rust API, JIT fusion and autotuning for speed.

People like these systems not because they are magical, but because they:

- Feel like normal programming.
- Support dynamic behavior (varying batch sizes, control flow).
- Still allow serious optimization for production workloads.

Bolt, as it stands, does not give that experience. The goal of this design is to fix that without throwing away the good parts of our existing ideas (IR, planning, arena allocation).

We want a framework that:

- Is **eager‑first** and **ergonomic**.
- Still has a strong **compiler and optimization** story when needed.
- Treats **dynamic shapes** and **control flow** as normal, not as special cases.
- Leans into **Rust’s strengths** (safety, ownership, concurrency), not just re‑implement PyTorch in another language.

---

## 2. High‑Level Philosophy

Let’s start with a simple sentence:

> **The eager API is the specification. Everything else is an optimization.**

This one line drives a lot of decisions:

- If a user calls `tensor.add(&other)`, that operation must behave correctly and predictably right now.
- If we later decide to compile a region of code that uses `add`, the behavior of that compiled version must match the eager semantics (up to numerical tolerances).
- If the compiler cannot handle a particular pattern (maybe due to dynamic control flow), we fall back to eager execution for that region. The user’s code must still work.

From this philosophy, a few principles follow:

1. There is only **one programming model**: Rust + eager `Tensor` operations.
2. The **IR and compiler** exist to accelerate portions of that eager program, not to replace it.
3. **Dynamic shapes** are handled at the eager level first; the compiler deals with them as best it can, but never forces the user into static‑shape straitjackets.
4. **Debuggability** is a feature, not an afterthought. A user should be able to reason about performance issues and correctness issues without learning our IR in depth.

Keep these in mind as you read the rest of the document. If any section seems to violate these principles, that is a smell we should discuss.

---

## 3. Big‑Picture Architecture Overview

We will describe the architecture in terms of layers. Each layer is a conceptual module, not necessarily a single crate or file.

Think of the stack like this:

1. **Eager Runtime**
   - User‑facing `Tensor` API.
   - Device abstraction and allocators.
   - Op dispatch (which kernel to run where).

2. **Tracing and Introspection**
   - Lightweight per‑thread trace logs of which ops are called, with shapes/dtypes/devices.
   - Optional: can be turned off in tight production loops.
   - Used for profiling, debugging, and as input to compilation.

3. **Region Compiler & IR**
   - Given traces (and/or explicit regions), construct a static graph (IR) for a particular function/region.
   - Lower IR to an executable plan (e.g., an arena‑planned program with `Instr`s).
   - Generate optimized kernels (fusion, vectorization, possibly via DSL or external codegen).

4. **Execution Engine for Compiled Regions**
   - Fast path: run compiled region with minimal overhead when shapes match a known signature.
   - Fallback path: call back into the eager runtime when shapes are different or features are unsupported.

We will walk through each of these layers in detail.

Before we do, let’s define some core concepts used throughout the design.

---

## 4. Core Concept: Tensor

We start with the most important type: `Tensor`.

### 4.1 What is a Tensor?

In our design, a `Tensor` is:

- A handle to device memory.
- A description of how to interpret that memory (shape, strides, dtype).
- A link to the device that “owns” that memory.
- Optionally, metadata for autograd.

It is **not**:

- A node in a global graph (in the eager path).
- A magical, lazy proxy that defers semantics until some unknown point in time.

In code, the core of `Tensor` looks like this:

```rust
use std::sync::Arc;

use crate::device::{Device, BufferId};
use crate::shape::ConcreteShape;
use crate::dtype::DType;

#[derive(Clone)]
pub struct BufferView {
    pub buffer_id: BufferId,
    pub offset_bytes: usize,
    pub dtype: DType,
    pub shape: ConcreteShape,
    pub strides: Vec<isize>,
}

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    view: BufferView,
    // Later: autograd metadata (grad_fn, requires_grad).
}

pub struct TensorStorage {
    runtime: Arc<Runtime>,
    device_kind: DeviceKind,
    buffer_id: BufferId,
}
```

Key updates from the earlier architecture:

- Every tensor knows its owning `Runtime`. That runtime owns the dispatcher, device map, and default device selection.
- `TensorStorage` no longer keeps an `Arc<dyn Device>` directly. Instead it stores `DeviceKind` and asks the runtime for the concrete device whenever we need to read/write/copy. This avoids reference cycles and keeps devices stateless.
- Drop is coordinated through the runtime: when storage is released we fetch the device from the runtime and free the buffer id. If the runtime is gone, we log loudly.

### 4.2 Basic Tensor Construction

Tensor allocation now flows through the runtime rather than ad-hoc device handles:

```rust
impl Runtime {
    pub fn tensor_from_slice<T: NativeType>(
        self: &Arc<Self>,
        shape: &[usize],
        data: &[T],
    ) -> Result<Tensor> {
        self.tensor_from_slice_on(self.default_device, shape, data)
    }

    pub fn tensor_from_slice_on<T: NativeType>(
        self: &Arc<Self>,
        device: DeviceKind,
        shape: &[usize],
        data: &[T],
    ) -> Result<Tensor> {
        Tensor::from_slice_in(self, device, shape, data)
    }
}
```

`Tensor::from_slice_in` performs the actual allocation, but the caller never touches a global singleton. You build a runtime via `Runtime::builder()`, register the devices/kernels you care about (e.g., `Runtime::builder().with_cpu()?.build()?`), and then use that runtime everywhere. Tests can spin up multiple runtimes side-by-side without interference, and advanced users can explicitly opt into `*_on` variants when they need a specific device rather than the runtime’s default.

### 4.3 Views and Non‑Allocating Ops

Operations like `reshape`, `transpose`, `slice` should not always allocate new memory. They can simply adjust the `shape` and `strides` in a new `Tensor`.

Example:

```rust
impl Tensor {
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let old_numel = self.view.shape.num_elements();
        let new_shape = ConcreteShape::from_slice(new_shape)?;
        let new_numel = new_shape.num_elements();
        if old_numel != new_numel {
            return Err(Error::InvalidShape("reshape: element count mismatch".into()));
        }

        Ok(Self {
            storage: self.storage.clone(),
            view: BufferView {
                buffer_id: self.view.buffer_id,
                offset_bytes: self.view.offset_bytes,
                dtype: self.view.dtype,
                shape: new_shape.clone(),
                strides: new_shape.contiguous_strides(),
            },
        })
    }
}
```

All tensor/layout entrypoints funnel through the same guard rails: `ConcreteShape::from_slice` (and the fallible `TryFrom<Vec<usize>>`) reject empty shapes or zero-sized dimensions. That keeps downstream layout math and kernel dispatch honest—there is no unchecked constructor that can smuggle invalid metadata into the runtime.

We avoid unnecessary allocations and data copies. This is standard practice in all serious frameworks and is crucial for performance.

---

## 5. Core Concept: Device and Allocator

Our `Tensor` depends on a `Device`. Let’s define what a `Device` is in this architecture.

### 5.1 Device Responsibilities

A `Device` is responsible for:

- Allocating and freeing device memory.
- Copying data between host and device.
- Copying data between device buffers.
- Synchronizing outstanding work (in simple configurations).
- Potentially exposing higher‑level primitives (streams, events) in the future.

In code:

```rust
pub trait Device: Send + Sync {
    fn target(&self) -> Target; // e.g., Cpu, Cuda, Metal

    fn alloc(&self, size: usize) -> Result<BufferId>;
    fn free(&self, buffer: BufferId);

    fn memcpy_h2d(&self, dst: BufferId, dst_offset: usize, src: &[u8]) -> Result<()>;
    fn memcpy_d2h(&self, src: BufferId, src_offset: usize, dst: &mut [u8]) -> Result<()>;
    fn memcpy_d2d(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        size: usize,
    ) -> Result<()>;

    fn sync(&self) -> Result<()>;
}
```

This is close to the current Bolt `Device` trait. The new design emphasizes that this trait is the base for both eager and compiled execution.

### 5.2 Allocator Abstraction (Optional but Important)

For more advanced memory management, we can introduce a separate `Allocator` trait that `Device` delegates to. This allows:

- Different allocation strategies (simple `Vec<u8>` on CPU, caching allocator on GPU).
- Better control over fragmentation and performance.

At first, we might bake a simple allocator directly into `Device`. Later, we can extract it cleanly once we know our patterns and needs.

### 5.3 Example: CpuDevice Sketch

Here is a highly simplified sketch of a CPU device implementation:

```rust
pub struct CpuDevice {
    buffers: Mutex<Slab<Vec<u8>>>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(Slab::new()),
        }
    }
}

impl Device for CpuDevice {
    fn target(&self) -> Target {
        Target::Cpu
    }

    fn alloc(&self, size: usize) -> Result<BufferId> {
        if size == 0 {
            return Err(Error::InvalidShape("buffer size must be > 0".into()));
        }
        let mut buffers = self.buffers.lock().unwrap();
        let key = buffers.insert(vec![0u8; size]);
        Ok(BufferId(key as u32))
    }

    fn free(&self, id: BufferId) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.contains(id.0 as usize) {
            buffers.remove(id.0 as usize);
        }
    }

    fn memcpy_h2d(&self, dst: BufferId, dst_offset: usize, src: &[u8]) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get_mut(dst.0 as usize)
            .ok_or_else(|| Error::InvalidBufferId(dst.0))?;
        if src.len() > buf.len() - dst_offset {
            return Err(Error::SizeMismatch {
                expected: buf.len() - dst_offset,
                actual: src.len(),
            });
        }
        buf[dst_offset..dst_offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn memcpy_d2h(&self, src: BufferId, src_offset: usize, dst: &mut [u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(src.0 as usize)
            .ok_or_else(|| Error::InvalidBufferId(src.0))?;
        if dst.len() > buf.len() - src_offset {
            return Err(Error::SizeMismatch {
                expected: buf.len() - src_offset,
                actual: dst.len(),
            });
        }
        dst.copy_from_slice(&buf[src_offset..src_offset + dst.len()]);
        Ok(())
    }

    fn memcpy_d2d(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;

        let (src_buf, dst_buf) = match buffers.get2_mut(src_idx, dst_idx) {
            Some(pair) => pair,
            None => return Err(Error::InvalidBufferId(src.0)),
        };

        if src_offset + size > src_buf.len() || dst_offset + size > dst_buf.len() {
            return Err(Error::SizeMismatch {
                expected: size,
                actual: std::cmp::min(
                    src_buf.len().saturating_sub(src_offset),
                    dst_buf.len().saturating_sub(dst_offset),
                ),
            });
        }

        dst_buf[dst_offset..dst_offset + size]
            .copy_from_slice(&src_buf[src_offset..src_offset + size]);

        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}
```

In the production `bolt-cpu` crate we now wrap `memcpy_d2d` (`Device::copy`) in a two-phase transfer: read the source range into a temporary `Vec<u8>`, drop the read lock, then write into the destination. A same-buffer fast path uses `copy_within` so overlapping ranges behave like `memmove`. This design choice prioritizes deadlock safety (multiple threads copying opposite directions can no longer wedge on nested `RwLock`s) while leaving room to revisit deterministic lock ordering plus chunked copies once profiling shows the method is performance-critical.

Because both `CpuDevice` and `BufferCell` are synchronized through `Mutex`/`RwLock`, we must treat lock poisoning as a first-class error rather than cascading panics. Any panic while `buffers.lock()` or `BufferCell::{read,write}` is held now returns `Error::DeviceLockPoisoned { device: DeviceKind::Cpu, lock: &'static str }` the next time that lock is acquired. Callers can inspect `Error::is_device_poisoned()` to short-circuit scheduling, and the human-readable `lock` labels (`"buffers"`, `"buffer_cell.read"`, `"buffer_cell.write"`) make it clear whether the failure is scoped to one buffer or the entire device map.

The important thing for the design is that **all higher‑level layers use `Device` as the abstraction boundary**. Whether we are in eager mode or running a compiled plan, we operate in terms of `BufferId`, `BufferView`, and `Device`.

---

## 6. Core Concept: Operation Dispatcher

We need a way to connect high‑level operations like `add`, `matmul`, `relu` to concrete implementations for specific devices and dtypes.

We implement this through a simple dispatcher.

### 6.1 OpKind and DeviceKind

We define some enums:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Add,
    MatMul,
    Relu,
    // more ops...
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    // more backends...
}
```

### 6.2 Dispatcher Table

The dispatcher holds a mapping from:

```text
(OpKind, DeviceKind, DType) -> implementation function
```

Sketch:

```rust
type EagerOpFn = dyn Fn(&[Tensor]) -> Result<Vec<Tensor>> + Send + Sync;

pub struct Dispatcher {
    table: HashMap<(OpKind, DeviceKind, DType), Arc<EagerOpFn>>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        op: OpKind,
        dev: DeviceKind,
        dtype: DType,
        f: Arc<EagerOpFn>,
    ) {
        self.table.insert((op, dev, dtype), f);
    }

    pub fn call(&self, op: OpKind, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // For now, assume all inputs are on same device and dtype.
        let dev_kind = inputs[0].device().kind();
        let dtype = inputs[0].dtype();

        let key = (op, dev_kind, dtype);
        let f = self
            .table
            .get(&key)
            .ok_or_else(|| Error::KernelError(format!("no kernel for {:?}", key)))?;

        f(inputs)
    }
}
```

The actual implementation will need more nuance (different dtypes per input, broadcasting rules, etc.), but this sketch shows the idea.

### 6.3 Tensor Methods Call Dispatcher

Now we wire this into `Tensor`:

```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_runtime(other)?;
        let runtime = self.runtime();
        runtime.dispatch_single(OpKind::Add, &[self.clone(), other.clone()], OpAttrs::None)
    }
}
```

For attribute-carrying ops we promote a tiny struct that implements an `Operation` trait. That struct (e.g., `SumOp { axes: Vec<usize> }`) knows how to convert to/from the erased `OpAttrs` enum, so the dispatcher enforces `OpKind`/attr pairing:

```rust
#[derive(Clone, Debug)]
pub struct SumOp {
    pub axes: Vec<usize>,
}

impl Operation for SumOp {
    const KIND: OpKind = OpKind::Sum;

    fn to_opattrs(&self) -> OpAttrs {
        OpAttrs::Sum(self.clone())
    }

    fn from_opattrs(attrs: &OpAttrs) -> Result<Self> {
        match attrs {
            OpAttrs::Sum(op) => Ok(op.clone()),
            other => Err(Error::OpAttrMismatch {
                op: OpKind::Sum,
                expected: "Sum",
                actual: other.discriminant_name(),
            }),
        }
    }
}

impl Tensor {
    pub fn sum_axes(&self, axes: &[usize]) -> Result<Tensor> {
        let axes = canonical_axes(axes, self.shape().len())?;
        let runtime = self.runtime();
        runtime.dispatch_op_single(&SumOp { axes }, &[self.clone()])
    }
}
```

Devices register kernels through `Dispatcher::register_operation::<SumOp, _>(...)`, so kernels receive a typed `&SumOp` instead of spelunking through a bag of attrs. This is the canonical representation future tracing/JIT layers will inspect.

Instead of hiding a dispatcher singleton somewhere in `bolt-core`, the runtime now *is* the execution context. Tests can create multiple runtimes in the same process, and tensors keep an `Arc<Runtime>` so we can call `Arc::ptr_eq` to reject mixed-runtime ops early. If you want global ergonomics, you build them as a façade on top of `Runtime`, not inside the core crates.

This is the core of the **eager execution path**. Every op:

- Validates arguments.
- Calls into appropriate kernel via the dispatcher.
- Returns new Tensors.

No global graph is mutated here. No VM is invoked. We are simply doing dynamic dispatch plus kernel calls.

---

## 7. Core Concept: Tracing

We want the ability to compile and optimize parts of a model. To do that, we need to know:

- What ops were executed?
- In what order?
- On which Tensors (shapes/dtypes/devices)?

We can answer this by **tracing**.

### 7.1 Trace Events

We define a simple `TraceEvent` type:

```rust
#[derive(Clone, Debug)]
pub enum TraceEvent {
    Op {
        op: OpKind,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        shapes: Vec<ConcreteShape>,
        dtypes: Vec<DType>,
        device: DeviceKind,
        // optionally: source location or call site info
    },
}
```

Here `TensorId` is a unique identifier for a tensor in the context of tracing. It might be assigned by a simple counter.

### 7.2 Trace Log

Each thread (or each “eager context”) can maintain a trace log:

```rust
pub struct TraceLog {
    events: Mutex<Vec<TraceEvent>>,
}

impl TraceLog {
    pub fn record(&self, event: TraceEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub fn drain(&self) -> Vec<TraceEvent> {
        let mut guard = self.events.lock().unwrap();
        let events = guard.clone();
        guard.clear();
        events
    }
}
```

We don’t want always‑on tracing in every deployment, because it adds overhead. So we make tracing:

- Configurable: can be enabled/disabled per session or per region.
- Cheap: events are simple, no complex graph mutations.

### 7.3 Eager Ops Emit Trace Events

When an eager op runs, it can optionally emit a trace event.

Example for `add`:

```rust
fn eager_add_impl(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    // actual implementation (kernel, allocation, etc.)
    let x = &inputs[0];
    let y = &inputs[1];

    // ... shape checks, allocation, device calls ...
    let out = /* ... */ ;

    // tracing
    if let Some(trace) = maybe_current_trace_log() {
        trace.record(TraceEvent::Op {
            op: OpKind::Add,
            inputs: vec![x.id(), y.id()],
            outputs: vec![out.id()],
            shapes: vec![x.shape().clone(), y.shape().clone(), out.shape().clone()],
            dtypes: vec![x.dtype(), y.dtype(), out.dtype()],
            device: x.device().kind(),
        });
    }

    Ok(vec![out])
}
```

This pattern gives us a **latent graph** of what actually happened during a function or training step, without affecting core semantics.

---

## 8. Region Compilation: From Trace to IR

Now we come to the compiler. The compiler’s job is to take a chunk of the trace log (representing some region of computation) and turn it into:

- A static IR (a graph or a linear program).
- An optimized execution plan (maybe with fused kernels and arena planning).

We will focus on the IR that looks very similar to Bolt’s existing `Program`.

### 8.1 What is a Region?

A region is simply:

- A set of ops and tensors that form a sub‑computation we want to optimize as a whole.

Typical regions:

- A module’s `forward` function.
- A single attention block.
- The body of a training step (forward + backward).

We can define regions in two ways:

1. **Explicit**: the user wraps a function with `bolt::region` or uses `bolt::compile`.
2. **Implicit**: we automatically detect hot code paths based on tracing and profile data.

We will start with explicit regions because they are easier to reason about.

### 8.2 Building a Graph from Trace Events

Given a list of `TraceEvent::Op`, we can build a graph:

- Nodes:
  - `Var`‑like entities representing intermediate tensors.
- Edges:
  - Ops (with OpKind, attributes) connecting input and output vars.

This is similar to Bolt’s existing `Graph` in `bolt-core`.

Sketch:

```rust
pub struct Graph {
    pub vars: Vec<Var>,
    pub ops: Vec<Op>,
}

pub struct Var {
    pub id: VarId,
    pub shape: ConcreteShape,
    pub dtype: DType,
    pub name: String, // optional
}

pub struct Op {
    pub id: OpId,
    pub kind: OpKind,
    pub inputs: Vec<VarId>,
    pub outputs: Vec<VarId>,
}
```

We can derive this Graph from trace by:

- Assigning a `VarId` to each unique `TensorId` encountered.
- For each `TraceEvent::Op`, creating an `Op` with appropriate input/output var IDs.

### 8.3 Lowering Graph to IR (Program)

Once we have a Graph, we can reuse a simplified version of Bolt’s `Program` concept:

```rust
pub struct Program {
    pub target: Target,
    pub instrs: Vec<Instr>,
    pub buffer_plan: BufferPlan,
}

pub enum Instr {
    Op {
        kernel: Arc<dyn Kernel>,
        inputs: Vec<BufId>,
        outputs: Vec<BufId>,
    },
}

pub struct BufferPlan {
    pub arenas: Vec<ArenaDesc>,
    pub buffers: HashMap<BufId, PlannedBuffer>,
}
```

Here:

- `BufId` is essentially a `VarId` from the Graph; we treat each var as being backed by some buffer.
- `Kernel` is a backend implementation chosen by a kernel registry (similar to dispatcher but for compiled mode).
- `BufferPlan` describes how we will allocate device memory for intermediate results using arenas.

This specific structure can be refined, but the key idea is:

> The IR is a **closed, static description** of how to execute the region, including how to manage memory.

### 8.4 Compiling IR to Optimized Code

There are multiple ways to execute `Program`:

1. **Interpretation**:
   - Use a VM that:
     - Allocates arenas based on `BufferPlan`.
     - Maps `BufId` to `BufferView`.
     - Runs each `Instr::Op` by calling `kernel.launch(device, &KernelLaunch)`.
   - This is basically your current Bolt VM.

2. **Code generation**:
   - Compile the IR to:
     - Specialized CPU code (Rust or C++).
     - GPU kernels (via a DSL like CubeCL or Triton).
   - This requires more engineering but can yield much better performance.

The important point: **the eager user does not need to know which path is used**. They simply see that a particular region is faster after the first run.

---

## 9. Executing Compiled Regions

Once we have a compiled `Program` (or equivalent), we want to:

- Run it instead of the eager path when possible.
- Fall back to eager when the shapes or conditions don’t match.

### 9.1 Shape Signatures and Guards

We define a **shape signature** for a compiled region:

```rust
pub struct ShapeSignature {
    pub input_shapes: Vec<ConcreteShape>,
    pub dtypes: Vec<DType>,
    pub device: DeviceKind,
}
```

When we compile a region, we store:

- The shape signature for which it is valid.
- The compiled `Program` (or code pointer).

At runtime:

- When we call the region again, we compute the current shape signature.
- If it matches a compiled one, we take the fast path.
- If it does not match, we either:
  - Compile a new variant for this signature (and cache it).
  - Or fall back to eager (depending on configuration).

### 9.2 Wiring into the User’s Code

We can model a region as a wrapper:

```rust
pub struct CompiledRegion<M> {
    module: M, // user-defined struct implementing Module
    cache: Mutex<HashMap<ShapeSignature, CompiledPlan>>,
}

pub struct CompiledPlan {
    program: Arc<Program>,
    // later: specialized machine code, kernel configs, etc.
}

impl<M: Module> CompiledRegion<M> {
    pub fn forward(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let sig = ShapeSignature::from_inputs(inputs)?;

        // Fast path: if we have a compiled plan.
        if let Some(plan) = self.cache.lock().unwrap().get(&sig) {
            return execute_program(&plan.program, inputs);
        }

        // Slow path: run eagerly and maybe compile.
        let outputs = self.module.forward_eager(inputs)?;

        // Optionally: trigger compilation and cache the plan.
        let trace = collect_trace_for_last_forward();
        let graph = graph_from_trace(trace)?;
        let program = compile_graph_to_program(graph)?;

        self.cache
            .lock()
            .unwrap()
            .insert(sig.clone(), CompiledPlan { program: Arc::new(program) });

        Ok(outputs)
    }
}
```

This is just a high‑level picture, but you can see how the compilation is:

- An _internal optimization_ of the eager forward pass.
- Invisible to the user beyond “first call is slower, subsequent calls are faster”.

---

## 10. Dynamic Shapes: Early Strategy vs Future Extensions

Dynamic shapes are a big topic, and we need to be honest about what we can do well now versus later.

### 10.1 Early Strategy: Per‑Shape Specialization

Initially, we will:

- Let eager execution handle **any** shapes.
- Let the compiler:
  - Specialize only for the shapes it has seen.
  - Cache compiled plans per shape signature.
- Not introduce symbolic dims in the IR yet.

This keeps the IR and compiler simple while still giving performance wins for common cases.

### 10.2 Later Strategy: Symbolic Batch Dimension

Once we have real users and workloads, we may find:

- Many regions only vary along batch dimension.

At that point we can:

- Extend our IR `Shape` to support a symbolic batch dim (e.g. `B`).
- Teach the compiler to:
  - Represent shapes like `[B, 512]`.
  - Plan memory in terms of `B`.
  - Generate code that loops over `B` at runtime.

We will _not_ attempt to generalize all dims to fully symbolic from day one. That is a massive project and easy to get wrong.

We move in stages, guided by real usage.

---

## 11. Control Flow: Practical Support

Control flow (if/loop based on tensor values) is often cited as a reason for preferring PyTorch over older TF. We want to support it well without over‑promising.

### 11.1 Eager Control Flow (Baseline)

Eager control flow “just works” because:

- Operations like `sum`, `mean`, `argmax` return scalar `Tensor`s.
- Methods like `.item::<f32>()` convert a scalar tensor to a Rust primitive.
- You can use those primitives in Rust `if`, `match`, and `loop` constructs.

Example:

```rust
fn my_model(x: Tensor, y: Tensor) -> Tensor {
    let threshold: f32 = x.sum().item().unwrap();
    if threshold > 0.0 {
        x.matmul(&y)
    } else {
        x.add(&y)
    }
}
```

This function is fully defined in eager semantics. It can be tested and debugged without thinking about any graphs.

### 11.2 Compiled Control Flow: Only Where It Makes Sense

When we try to compile such a function:

- We may or may not be able to convert the control flow into graph‑level control flow.
- For a first version of the compiler, we do not attempt to compile heavy control‑flow logic. Instead:
  - We treat `.item()` as a **compile barrier**.
  - Everything before `.item()` can be part of a compiled region.
  - Everything after `.item()` can be another region or stay eager.

Later, if and when we want more aggressive control‑flow compilation, we can:

- Introduce a separate `#[bolt::compile]` attribute for advanced users.
- Restrict it to a subset of Rust that we know how to lower reliably to graph control flow.
- Document the limitations clearly.

The key idea: we avoid promising “arbitrary Rust control flow compiled to a graph” up front. We support eager control flow fully and only incrementally add more compiled support as we can do so _reliably_.

---

## 12. Backends and Kernels

Performance in a DL framework is dominated by:

- Kernel quality (for matmul, convolutions, normalizations, softmax, etc.).
- Memory behavior (allocation patterns, caching).

Our arch needs to make kernel integration straightforward.

### 12.1 Reference Kernels

We maintain a `bolt-kernels-reference` crate with:

- Simple, correctness‑first CPU kernels.
- No heavy vectorization, no fancy tiling.
- Used for:
  - Testing.
  - Fallback when optimized kernels are unavailable.

Example (add for CPU, F32, contiguous):

```rust
fn cpu_add_f32_contiguous(
    device: &Arc<dyn Device>,
    a: &BufferView,
    b: &BufferView,
    c: &BufferView,
) -> Result<()> {
    let numel = c.shape.num_elements();
    let size = numel * DType::F32.size_in_bytes();

    let mut a_bytes = vec![0u8; size];
    let mut b_bytes = vec![0u8; size];
    let mut c_bytes = vec![0u8; size];

    device.memcpy_d2h(a.buffer_id, a.offset_bytes, &mut a_bytes)?;
    device.memcpy_d2h(b.buffer_id, b.offset_bytes, &mut b_bytes)?;

    let a_f32 = bytemuck::cast_slice::<u8, f32>(&a_bytes);
    let b_f32 = bytemuck::cast_slice::<u8, f32>(&b_bytes);
    let c_f32 = bytemuck::cast_slice_mut::<u8, f32>(&mut c_bytes);

    for i in 0..numel {
        c_f32[i] = a_f32[i] + b_f32[i];
    }

    device.memcpy_h2d(c.buffer_id, c.offset_bytes, &c_bytes)
}
```

This is not meant to be fast; it is meant to be correct and simple.

### 12.2 Optimized Kernels

We then allow more sophisticated backends to register optimized kernels:

- CPU:
  - Use BLAS libraries for matmul.
  - Use explicit SIMD for elementwise ops.
  - Use cache‑friendly tiling for convolutions.
- GPU:
  - Use a DSL (CubeCL, Triton‑like) to generate and autotune kernels.
  - Integrate vendor libraries (cuBLAS, cuDNN) where appropriate.

The dispatcher (Section 6) chooses between reference and optimized kernels based on:

- DeviceKind.
- DType.
- Config (debug vs performance mode).

This separation lets us evolve kernels independently from the core runtime and compiler.

---

## 13. Tools and Developer Experience

It’s not enough to have a good execution model. Users need to be able to:

- Understand performance.
- Debug correctness issues.
- Move from research to production without rewriting everything.

### 13.1 Flight Recorder: Step‑Level Debugging

We can integrate a “flight recorder” mode:

- For a training step, we record:
  - Which ops were executed.
  - Shapes, dtypes, min/max values of tensors at certain checkpoints.
  - Whether each region ran eager or compiled.
- If a step fails (NaN, divergence, panic), we can:
  - Replay that step with all regions forced to eager.
  - Or force specific regions off the compiled path.

This is much more actionable than “RuntimeError somewhere in your compiled graph”.

### 13.2 Correctness Checks

We can offer APIs like:

```rust
bolt::check_grad(&model, &input, &target)?;
```

This would:

- Run a small gradient check using finite differences for a subset of parameters.
- Verify that backward computations are close to numerical derivatives.

We can also allow optional shape/dtype assertions:

```rust
#[bolt::expect_shape(batch, 512)]
fn forward(&self, x: Tensor) -> Tensor { ... }
```

At runtime, we can check that `x.shape()[1] == 512` and fail early with a clear error if not.

These tools build user confidence and reduce subtle bugs.

---

## 14. Summary of Key Decisions

Let’s list the most important decisions made in this design and why.

1. **Eager is the specification.**
   - Users write and debug in eager mode.
   - Compiled regions are accelerators built on top of eager semantics.

2. **No global graph drives execution.**
   - We do not base execution on an always‑active UCG.
   - Instead, we use a simple dispatcher and direct kernel calls.
   - We build graphs only when needed for compilation or analysis.

3. **Tracing as a service, not as a burden.**
   - Trace logs are optional and cheap.
   - They are inputs to the compiler and profiler, not a permanent runtime structure.

4. **Region‑based compilation with shape specialization.**
   - We compile specific regions of code, usually module `forward`s.
   - We specialize for concrete shapes and cache compiled plans.
   - We can add symbolic dims later where it pays off.

5. **Dynamic shapes are normal, not exotic.**
   - Eager supports arbitrary shapes by design.
   - Compilation adapts when possible; otherwise we fall back gracefully.

6. **Control flow is fully supported in eager.**
   - Plain Rust control flow “just works”.
   - Compiled control flow is added cautiously, based on what we can support reliably.

7. **Device and allocator abstractions are central.**
   - Everything talks in terms of `Device`, `BufferId`, `BufferView`.
   - Memory behavior is explicit and testable.

8. **Kernel quality is a first‑class concern.**
   - Reference kernels for correctness.
   - Optimized backends for performance.
   - A clear registry / dispatcher connecting high‑level ops to low‑level implementations.

9. **Tooling and introspection are baked in, not bolted on.**
   - Flight recorder.
   - Grad checks.
   - Shape/dtype expectations.
   - Region compile inspection.

---

## 15. Implementation Roadmap (High Level)

This document is already very long, so we’ll keep this section high‑level. The actual step plan can live in a separate `plan/` RFC.

**Phase 1: Eager Runtime**

- Implement the new `Tensor` struct and methods for:
  - `zeros`, `from_slice_f32`, `add`, `matmul`, `relu`, etc.
- Implement `Device` and `CpuDevice` as described.
- Implement a basic dispatcher and reference CPU kernels.
- Make sure we can run simple models end‑to‑end in eager mode.

**Phase 2: Tracing and Debug Tools**

- Implement `TraceLog` and `TraceEvent`.
- Add optional tracing instrumentation to eager ops.
- Implement simple tools to:
  - Dump traces.
  - Visualize op sequences.

**Phase 3: Region Compiler and IR**

- Build a Graph from traces.
- Implement a simplified `Program` IR and a basic compiler pass:
  - No heavy optimizations yet.
  - Interpret IR via a VM similar to current Bolt’s VM.

**Phase 4: Region Wrapper and Shape Caching**

- Implement `CompiledRegion` wrapper around `Module`.
- Add shape signature computation and per‑shape caching.
- Wire `bolt::compile(model)` to produce a wrapped model with compiled regions.

**Phase 5: Performance and Backends**

- Introduce optimized CPU kernels.
- Explore a GPU backend with a simple DSL and autotuning.
- Profile real workloads and prioritize ops/regions for optimization.

**Phase 6: Dynamic Shape Enhancements (Optional)**

- If needed, add symbolic batch dims to IR.
- Teach compiler and `Program` to be batch‑parametric.

**Phase 7: Advanced Control Flow and Autograd**

- Build dynamic autograd on top of eager ops.
- Add limited control‑flow compilation support if there is real demand.

---

## 16. Closing Thoughts

This design is intentionally ambitious in some places and conservative in others.

It is ambitious in that:

- It aims to give Bolt a PyTorch‑like experience in Rust.
- It takes compilation seriously, but as an internal service rather than the main programming model.
- It treats performance and tooling as first‑class concerns, not as later add‑ons.

It is conservative in that:

- It avoids over‑promising on full symbolic dynamic shapes and arbitrary control‑flow compilation from day one.
- It leans on simple, testable abstractions (`Device`, `Tensor`, `BufferView`, `Program`) rather than clever type‑level tricks.
- It respects the fact that good kernels and memory allocators matter more than fancy IRs.

If we get this right, we end up with a framework that:

- Feels natural to use for day‑to‑day deep learning work.
- Can be pushed hard in production settings.
- Is understandable and maintainable by a small team.

If we get it wrong, we at least want our mistakes to be **in the open**, well‑documented, and easy to discuss. This document is meant to be the starting point of that conversation, not the final word.
