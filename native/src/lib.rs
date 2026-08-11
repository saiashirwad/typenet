use candle_core::{DType, Device, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn to_napi_err(err: candle_core::Error) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

// ---------------------------------------------------------------------------
// Device selection: Metal on macOS when available, CPU otherwise.
// ---------------------------------------------------------------------------

fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return device;
            }
        }
        Device::Cpu
    })
}

#[napi]
pub fn device_name() -> String {
    match device() {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
        Device::Metal(_) => "metal".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Graph format: a topological list of nodes; inputs reference earlier
// indices; the last node is the root. Leaves index into the `leaves`
// Float32Array as contiguous slices of prod(shape) f32 values.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum Node {
    Leaf {
        leaf: usize,
        offset: usize,
        shape: Vec<usize>,
    },
    Binary {
        kind: String,
        parameter: f64,
        a: usize,
        b: usize,
        shape: Vec<usize>,
    },
    Unary {
        kind: String,
        parameter: f64,
        input: usize,
    },
    Matmul {
        a: usize,
        b: usize,
    },
    Reduce {
        kind: String,
        dim: usize,
        keepdim: bool,
        input: usize,
    },
    ReduceAll {
        kind: String,
        input: usize,
    },
    BroadcastTo {
        input: usize,
        shape: Vec<usize>,
    },
    Permute {
        order: Vec<usize>,
        input: usize,
    },
    View {
        input: usize,
        shape: Vec<usize>,
    },
    Narrow {
        dim: usize,
        start: usize,
        length: usize,
        input: usize,
    },
    Cat {
        a: usize,
        b: usize,
        dim: usize,
    },
    OneHot {
        classes: usize,
        input: usize,
    },
    /// Gather rows: out[j] = input[index[j]] along `dim`.
    IndexSelect {
        dim: usize,
        input: usize,
        index: usize,
    },
    /// Scatter-add rows into a zero tensor of `length` rows along `dim`:
    /// out[index[j]] += input[j].
    ScatterAdd {
        dim: usize,
        length: usize,
        input: usize,
        index: usize,
    },
    /// Random values, drawn fresh on every evaluation from a hash of
    /// (eval seed, stream, element index) — see `random_data`.
    Random {
        kind: String,
        stream: u32,
        shape: Vec<usize>,
    },
}

#[derive(Debug, Deserialize)]
struct Graph {
    nodes: Vec<Node>,
    /// Indices of the output nodes. Defaults to the last node, so
    /// single-root graphs from older callers keep working.
    #[serde(default)]
    roots: Option<Vec<usize>>,
    /// Which evaluator the JS side picked for this graph — see `Target`.
    #[serde(default)]
    device: Option<String>,
}

/// Where a graph runs. The JS side chooses (see pickTarget in
/// src/tensor.ts) because it knows the graph's total size before
/// anything crosses the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    /// The fused loop evaluator: no candle, no dispatch, no BLAS. Wins
    /// below a few tens of thousands of elements, where a kernel launch
    /// costs more than the arithmetic.
    Loops,
    /// candle on the CPU device, which on macOS means Accelerate for
    /// matmul. The default above the loop evaluator's range.
    Cpu,
    /// candle on the best accelerator (Metal where available).
    Accelerator,
}

impl Target {
    fn parse(hint: Option<&str>) -> Self {
        match hint {
            Some("loops") => Target::Loops,
            Some("gpu") => Target::Accelerator,
            _ => Target::Cpu,
        }
    }
}

fn prod(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Indices of the nodes a node directly reads.
fn node_inputs(node: &Node) -> Vec<usize> {
    match node {
        Node::Leaf { .. } => vec![],
        Node::Binary { a, b, .. } => vec![*a, *b],
        Node::Unary { input, .. } => vec![*input],
        Node::Matmul { a, b } => vec![*a, *b],
        Node::Reduce { input, .. } => vec![*input],
        Node::ReduceAll { input, .. } => vec![*input],
        Node::BroadcastTo { input, .. } => vec![*input],
        Node::Permute { input, .. } => vec![*input],
        Node::View { input, .. } => vec![*input],
        Node::Narrow { input, .. } => vec![*input],
        Node::Cat { a, b, .. } => vec![*a, *b],
        Node::OneHot { input, .. } => vec![*input],
        Node::IndexSelect { input, index, .. } => vec![*input, *index],
        Node::ScatterAdd { input, index, .. } => vec![*input, *index],
        Node::Random { .. } => vec![],
    }
}

/// Plain broadcast of two dim lists (align right, max-or-error), no tensors.
fn broadcast_dim_vecs(a: &[usize], b: &[usize]) -> candle_core::Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for j in 0..rank {
        let ad = if j < rank - a.len() { 1 } else { a[j - (rank - a.len())] };
        let bd = if j < rank - b.len() { 1 } else { b[j - (rank - b.len())] };
        if ad != bd && ad != 1 && bd != 1 {
            return Err(candle_core::Error::Msg(format!(
                "shapes {a:?} and {b:?} are not broadcastable"
            )));
        }
        out[j] = ad.max(bd);
    }
    Ok(out)
}

/// Output shape of every node, derived from the same shape math the JS
/// side used — no data touched.
fn node_shapes(graph: &Graph) -> candle_core::Result<Vec<Vec<usize>>> {
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let shape = match node {
            Node::Leaf { shape, .. } => shape.clone(),
            Node::Binary { shape, .. } => shape.clone(),
            Node::Unary { input, .. } => shapes[*input].clone(),
            Node::Matmul { a, b } => {
                let (sa, sb) = (&shapes[*a], &shapes[*b]);
                let (ar, br) = (sa.len(), sb.len());
                let mut out = broadcast_dim_vecs(&sa[..ar - 2], &sb[..br - 2])?;
                out.push(sa[ar - 2]);
                out.push(sb[br - 1]);
                out
            }
            Node::Reduce {
                dim,
                keepdim,
                input,
                ..
            } => {
                let mut s = shapes[*input].clone();
                if *keepdim {
                    s[*dim] = 1;
                } else {
                    s.remove(*dim);
                }
                s
            }
            Node::ReduceAll { .. } => vec![],
            Node::BroadcastTo { shape, .. } => shape.clone(),
            Node::Permute { order, input } => {
                order.iter().map(|&d| shapes[*input][d]).collect()
            }
            Node::View { shape, .. } => shape.clone(),
            Node::Narrow {
                dim,
                length,
                input,
                ..
            } => {
                let mut s = shapes[*input].clone();
                s[*dim] = *length;
                s
            }
            Node::Cat { a, b, dim } => {
                let mut s = shapes[*a].clone();
                s[*dim] += shapes[*b][*dim];
                s
            }
            Node::OneHot { classes, input } => vec![prod(&shapes[*input]), *classes],
            Node::IndexSelect { dim, input, index } => {
                let mut s = shapes[*input].clone();
                s[*dim] = prod(&shapes[*index]);
                s
            }
            Node::ScatterAdd {
                dim, length, input, ..
            } => {
                let mut s = shapes[*input].clone();
                s[*dim] = *length;
                s
            }
            Node::Random { shape, .. } => shape.clone(),
        };
        shapes.push(shape);
    }
    Ok(shapes)
}

// ---------------------------------------------------------------------------
// Counter-based random numbers. Element `i` of stream `s` under seed `k`
// is a pure hash of (k, s, i): no state to thread through the evaluator,
// every element independent, and the same arithmetic as the TS side
// (hash32 / unitFloat in src/tensor.ts). Uniform draws therefore match
// exactly across paths — integer mixing and an exact power-of-two scale;
// normal draws match to f32 rounding, since ln and cos are only
// specified that closely. The seed is an argument of the eval call, not
// part of the graph JSON, so a replayed compiled graph keeps its plan.
// ---------------------------------------------------------------------------

/// murmur3's 32-bit finalizer, Stafford 13 variant.
#[inline]
fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Uniform in [0, 1) from 24 mantissa bits of a hashed counter.
#[inline]
fn unit_float(seed: u32, stream: u32, i: u32) -> f32 {
    let mixed = hash32(hash32(seed ^ stream.wrapping_mul(0x9e37_79b9)) ^ i);
    (mixed >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn random_data(kind: &str, n: usize, stream: u32, seed: u32) -> candle_core::Result<Vec<f32>> {
    match kind {
        "uniform" => Ok((0..n).map(|i| unit_float(seed, stream, i as u32)).collect()),
        // Box-Muller per element from two independent draws, so element i
        // does not depend on how many were drawn before it.
        "normal" => Ok((0..n)
            .map(|i| {
                // f64 transcendentals, like the JS side, so the two stay
                // within f32 rounding of each other.
                let u = 1.0 - unit_float(seed, stream, 2 * i as u32) as f64;
                let v = unit_float(seed, stream, 2 * i as u32 + 1) as f64;
                ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
            })
            .collect()),
        other => Err(candle_core::Error::Msg(format!(
            "unknown random kind: {other}"
        ))),
    }
}

fn is_elementwise(node: &Node) -> bool {
    matches!(node, Node::Binary { .. } | Node::Unary { .. })
}

/// Broadcast both operands to their common shape so elementwise
/// (non-broadcast-aware) kernels work on identical layouts.
fn broadcast_pair(a: &Tensor, b: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
    let shape = a.shape().broadcast_shape_binary_op(b.shape(), "binary")?;
    Ok((a.broadcast_as(&shape)?, b.broadcast_as(shape)?))
}

fn elementwise(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    let (a, b) = broadcast_pair(a, b)?;
    f(&a.contiguous()?, &b.contiguous()?)
}

/// Candle comparisons return U8 masks; cast to F32 for arithmetic.
fn mask_f32(t: &Tensor) -> candle_core::Result<Tensor> {
    t.to_dtype(DType::F32)
}

fn eval_binary(kind: &str, parameter: f64, a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    match kind {
        "add" => a.broadcast_add(b),
        "sub" => a.broadcast_sub(b),
        "mul" => a.broadcast_mul(b),
        "div" => a.broadcast_div(b),
        "maximum" => elementwise(a, b, |x, y| x.maximum(y)),
        "minimum" => elementwise(a, b, |x, y| x.minimum(y)),
        "gt" => elementwise(a, b, |x, y| mask_f32(&x.gt(y)?)),
        "ge" => elementwise(a, b, |x, y| mask_f32(&x.ge(y)?)),
        "lt" => elementwise(a, b, |x, y| mask_f32(&x.lt(y)?)),
        "le" => elementwise(a, b, |x, y| mask_f32(&x.le(y)?)),
        "eq" => elementwise(a, b, |x, y| mask_f32(&x.eq(y)?)),
        "negDiv" => elementwise(a, b, |x, y| x.neg()? / y),
        "halfDiv" => elementwise(a, b, |x, y| (x * 0.5)? / y),
        "mulSign" => elementwise(a, b, |x, y| x * &y.sign()?),
        // sign(y).relu() is 1 where y > 0 and 0 elsewhere, without the
        // separate comparison and dtype cast a mask would need.
        "reluGrad" => elementwise(a, b, |x, y| x * &y.sign()?.relu()?),
        "leakyReluGrad" => elementwise(a, b, |x, y| {
            // where y > 0: x, else parameter * x  ==  x * (m + (1-m)*p)
            let m = mask_f32(&y.gt(&y.zeros_like()?)?)?;
            let weights = (&m + &(m.ones_like()? - &m)? * parameter)?;
            x * &weights
        }),
        // affine(-1, 1) is 1 - y in one kernel, with nothing allocated.
        "sigmoidGrad" => elementwise(a, b, |x, y| (x * y)? * &y.affine(-1.0, 1.0)?),
        "tanhGrad" => elementwise(a, b, |x, y| x * &y.sqr()?.affine(-1.0, 1.0)?),
        other => Err(candle_core::Error::Msg(format!(
            "unknown binary op: {other}"
        ))),
    }
}

fn eval_unary(kind: &str, parameter: f64, a: &Tensor) -> candle_core::Result<Tensor> {
    match kind {
        "pow" => a.powf(parameter),
        "neg" => a.neg(),
        "exp" => a.exp(),
        "log" => a.log(),
        "sqrt" => a.sqrt(),
        "abs" => a.abs(),
        "relu" => a.relu(),
        // relu(x) - p*relu(-x)
        "leakyRelu" => a.relu()? - (a.neg()?.relu()? * parameter)?,
        // sigmoid(x) = (tanh(x/2) + 1)/2. Three kernels against the five
        // that 1/(1 + exp(-x)) needs here, and it does not overflow for
        // large negative x either.
        "sigmoid" => a.affine(0.5, 0.0)?.tanh()?.affine(0.5, 0.5),
        "tanh" => a.tanh(),
        "scalePowGrad" => a.powf(parameter - 1.0)? * parameter,
        other => Err(candle_core::Error::Msg(format!(
            "unknown unary op: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Scalar evaluation of the elementwise ops (one f32 in, one f32 out). Used
// by the CPU fusion pass; must mirror eval_binary / eval_unary exactly.
// ---------------------------------------------------------------------------

/// Elementwise op kinds, resolved from their JSON names once when a graph
/// is prepared. The evaluator's inner loop runs per element, so matching
/// on a string there — which is what it used to do — cost more than the
/// arithmetic it was dispatching.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    NegDiv,
    HalfDiv,
    MulSign,
    ReluGrad,
    LeakyReluGrad,
    SigmoidGrad,
    TanhGrad,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Un {
    Pow,
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    ScalePowGrad,
}

impl Bin {
    fn parse(kind: &str) -> candle_core::Result<Self> {
        Ok(match kind {
            "add" => Bin::Add,
            "sub" => Bin::Sub,
            "mul" => Bin::Mul,
            "div" => Bin::Div,
            "maximum" => Bin::Maximum,
            "minimum" => Bin::Minimum,
            "gt" => Bin::Gt,
            "ge" => Bin::Ge,
            "lt" => Bin::Lt,
            "le" => Bin::Le,
            "eq" => Bin::Eq,
            "negDiv" => Bin::NegDiv,
            "halfDiv" => Bin::HalfDiv,
            "mulSign" => Bin::MulSign,
            "reluGrad" => Bin::ReluGrad,
            "leakyReluGrad" => Bin::LeakyReluGrad,
            "sigmoidGrad" => Bin::SigmoidGrad,
            "tanhGrad" => Bin::TanhGrad,
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown binary op: {other}"
                )))
            }
        })
    }
}

impl Un {
    fn parse(kind: &str) -> candle_core::Result<Self> {
        Ok(match kind {
            "pow" => Un::Pow,
            "neg" => Un::Neg,
            "exp" => Un::Exp,
            "log" => Un::Log,
            "sqrt" => Un::Sqrt,
            "abs" => Un::Abs,
            "relu" => Un::Relu,
            "leakyRelu" => Un::LeakyRelu,
            "sigmoid" => Un::Sigmoid,
            "tanh" => Un::Tanh,
            "scalePowGrad" => Un::ScalePowGrad,
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown unary op: {other}"
                )))
            }
        })
    }
}

/// One resolved elementwise operation: which op, and its scalar parameter
/// (the exponent of `pow`, the slope of `leakyRelu`).
#[derive(Clone, Copy)]
enum Op {
    Bin(Bin, f32),
    Un(Un, f32),
}

impl Op {
    fn of(node: &Node) -> candle_core::Result<Self> {
        match node {
            Node::Binary {
                kind, parameter, ..
            } => Ok(Op::Bin(Bin::parse(kind)?, *parameter as f32)),
            Node::Unary {
                kind, parameter, ..
            } => Ok(Op::Un(Un::parse(kind)?, *parameter as f32)),
            _ => Err(candle_core::Error::Msg(
                "elementwise plans only contain elementwise nodes".into(),
            )),
        }
    }
}

#[inline(always)]
fn apply_bin(kind: Bin, p: f32, a: f32, b: f32) -> f32 {
    match kind {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        // f32::max/min return the non-NaN operand; candle and JS both
        // propagate NaN, so compare explicitly.
        Bin::Maximum => {
            if a >= b {
                a
            } else {
                b
            }
        }
        Bin::Minimum => {
            if a <= b {
                a
            } else {
                b
            }
        }
        Bin::Gt => (a > b) as u8 as f32,
        Bin::Ge => (a >= b) as u8 as f32,
        Bin::Lt => (a < b) as u8 as f32,
        Bin::Le => (a <= b) as u8 as f32,
        Bin::Eq => (a == b) as u8 as f32,
        Bin::NegDiv => -a / b,
        Bin::HalfDiv => 0.5 * a / b,
        Bin::MulSign => a * ((b > 0.0) as u8 as f32 - (b < 0.0) as u8 as f32),
        Bin::ReluGrad => {
            if b > 0.0 {
                a
            } else {
                0.0
            }
        }
        Bin::LeakyReluGrad => a * if b > 0.0 { 1.0 } else { p },
        Bin::SigmoidGrad => a * b * (1.0 - b),
        Bin::TanhGrad => a * (1.0 - b * b),
    }
}

#[inline(always)]
fn apply_un(kind: Un, p: f32, x: f32) -> f32 {
    match kind {
        Un::Pow => x.powf(p),
        Un::Neg => -x,
        Un::Exp => x.exp(),
        Un::Log => x.ln(),
        Un::Sqrt => x.sqrt(),
        Un::Abs => x.abs(),
        Un::Relu => x.max(0.0),
        Un::LeakyRelu => {
            if x > 0.0 {
                x
            } else {
                p * x
            }
        }
        Un::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Un::Tanh => x.tanh(),
        Un::ScalePowGrad => p * x.powf(p - 1.0),
    }
}

// ---------------------------------------------------------------------------
// Tiny-graph CPU evaluator. Graphs the JS side pins with `device: "cpu"`
// (≤ CPU_HINT_MAX_WORK total elements) are dominated by candle's per-op
// dispatch — ~0.7µs per kernel × ~60 kernels for a small training step,
// while the FFI hop itself is ~0.8µs. For those we skip candle entirely
// and evaluate the graph directly on Vec<f32> buffers with plain loops,
// fusing maximal chains of elementwise (binary/unary) nodes into single
// passes: no per-op dispatch, no intermediate tensors.
//
// Fusion absorption rule (always correct, never recomputes): a node joins
// its consumer's group only if it is elementwise, live, broadcastable to
// the group output shape, has exactly that one consumer, and is not a
// root. Members with the output shape are evaluated together in one
// scratch pass; members with smaller shapes are evaluated first into temp
// buffers (their inputs can only be external nodes or smaller members
// still), so no value is ever computed twice. Barrier ops (matmul,
// reduce, permute, cat, narrow, oneHot) run as plain loops over Vecs —
// all tiny by construction.
// ---------------------------------------------------------------------------

struct FusionPlan {
    /// group id per node (members and leader), or None.
    group_of: Vec<Option<usize>>,
    /// members per group, topological order; the leader is the last entry.
    groups: Vec<Vec<usize>>,
}

fn broadcastable_to(shape: &[usize], out: &[usize]) -> bool {
    if shape.len() > out.len() {
        return false;
    }
    let offset = out.len() - shape.len();
    (0..shape.len()).all(|j| shape[j] == 1 || shape[j] == out[offset + j])
}

fn plan_fusion(
    graph: &Graph,
    shapes: &[Vec<usize>],
    live: &[bool],
    is_root: &[bool],
) -> FusionPlan {
    let n = graph.nodes.len();
    // Consumer counts over live edges only.
    let mut consumers = vec![0usize; n];
    for (i, node) in graph.nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        for input in node_inputs(node) {
            consumers[input] += 1;
        }
    }
    let mut group_of: Vec<Option<usize>> = vec![None; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    // Reverse topo order: consumers get to be leaders before their inputs
    // are claimed by someone else.
    for leader in (0..n).rev() {
        if !live[leader]
            || !is_elementwise(&graph.nodes[leader])
            || group_of[leader].is_some()
        {
            continue;
        }
        let mut members = vec![leader];
        group_of[leader] = Some(groups.len());
        let mut stack = vec![leader];
        while let Some(m) = stack.pop() {
            for c in node_inputs(&graph.nodes[m]) {
                if live[c]
                    && is_elementwise(&graph.nodes[c])
                    && group_of[c].is_none()
                    && !is_root[c]
                    && consumers[c] == 1
                    && broadcastable_to(&shapes[c], &shapes[leader])
                {
                    group_of[c] = Some(groups.len());
                    members.push(c);
                    stack.push(c);
                }
            }
        }
        if members.len() == 1 {
            // A singleton group buys nothing over the plain candle path.
            group_of[leader] = None;
        } else {
            members.sort_unstable();
            groups.push(members);
        }
    }
    FusionPlan { group_of, groups }
}

/// Row-major strides of `shape` aligned against `out_shape` (broadcast:
/// stride 0 on size-1 or missing-leading dims), for flat-index mapping.
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let offset = rank - shape.len();
    let mut strides = vec![0usize; rank];
    let mut stride = 1usize;
    for j in (0..shape.len()).rev() {
        if shape[j] != 1 {
            strides[offset + j] = stride;
        }
        stride *= shape[j];
    }
    strides
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for j in (0..shape.len()).rev() {
        strides[j] = stride;
        stride *= shape[j];
    }
    strides
}

fn flat_to_coords(mut i: usize, shape: &[usize], coords: &mut [usize]) {
    for j in (0..shape.len()).rev() {
        coords[j] = i % shape[j];
        i /= shape[j];
    }
}

/// Read element `i` (flat, in `out_shape` space) of a buffer whose own
/// shape broadcasts to `out_shape` via `strides`.
#[inline]
fn read_bcast(data: &[f32], strides: &[usize], same_shape: bool, i: usize, coords: &[usize]) -> f32 {
    if same_shape {
        return data[i];
    }
    if data.len() == 1 {
        return data[0];
    }
    let mut idx = 0usize;
    for j in 0..strides.len() {
        idx += coords[j] * strides[j];
    }
    data[idx]
}

// ---------------------------------------------------------------------------
// Prepared plans: everything derivable from the graph JSON (shapes,
// liveness, fusion groups, broadcast strides, child resolution) is computed
// ONCE and cached — compile() replays the same JSON hundreds of times, and
// on cache hits a step is just leaf copies + raw loops + readback.
// ---------------------------------------------------------------------------

/// Where a fused-pass input reads from: a graph buffer, a small-member
/// temp, or a scratch slot (a same-shape member evaluated in this pass).
enum ChildSource {
    Buffer(usize),
    Temp(usize),
    Slot(usize),
}

struct ChildRef {
    source: ChildSource,
    /// Broadcast strides of the child's shape against the pass's target
    /// shape (empty for Slot children, which are always same-shape).
    strides: Vec<usize>,
    same_shape: bool,
}

/// One elementwise operation with its inputs fully resolved: a fused
/// group's member, or a standalone node that fusion left on its own.
struct MemberPlan {
    /// The resolved elementwise operation.
    op: Op,
    /// Shape this pass produces (the group output shape for main members;
    /// the member's own smaller shape for small members).
    out_shape: Vec<usize>,
    /// Fully resolved inputs (1 for unary, 2 for binary).
    inputs: Vec<ChildRef>,
    /// Every input already has the output shape, so the pass needs no
    /// coordinate arithmetic.
    all_same: bool,
}

struct GroupPlan {
    leader: usize,
    out_shape: Vec<usize>,
    /// True when no member reads a broadcast input, so the pass can index
    /// buffers directly instead of decomposing a flat index into coords.
    all_same: bool,
    /// Members smaller than the output shape, topo order; temp index =
    /// position. Their inputs can only be Buffer or earlier Temps.
    small_members: Vec<MemberPlan>,
    /// Same-shape-as-output members, topo order; scratch slot = position;
    /// the leader is last.
    main_members: Vec<MemberPlan>,
}

struct PreparedGraph {
    graph: Graph,
    shapes: Vec<Vec<usize>>,
    roots: Vec<usize>,
    live: Vec<bool>,
    /// How many live nodes read each node. Both evaluators count down as
    /// they go and drop a buffer once nothing else will read it, which is
    /// what keeps a long rollout from holding every activation it ever
    /// produced — and, on Metal, what lets candle's allocator hand the
    /// same device buffers back out instead of asking for new ones.
    consumers: Vec<usize>,
    /// True for nodes whose value is returned, so they are never dropped.
    is_root: Vec<bool>,
    /// group index per member node (skip during the main loop); leaders
    /// trigger execution.
    group_of: Vec<Option<usize>>,
    groups: Vec<GroupPlan>,
    /// Per-node plans for standalone elementwise nodes.
    ewise: Vec<Option<MemberPlan>>,
    /// Which evaluator this graph runs on, chosen by the JS side.
    target: Target,
}

impl PreparedGraph {
    fn prepare(graph: Graph) -> candle_core::Result<Self> {
        let shapes = node_shapes(&graph)?;
        let n = graph.nodes.len();
        let roots: Vec<usize> = match &graph.roots {
            Some(roots) => roots.clone(),
            None => vec![n.saturating_sub(1)],
        };
        let mut live = vec![false; n];
        let mut stack = roots.clone();
        while let Some(i) = stack.pop() {
            if live[i] {
                continue;
            }
            live[i] = true;
            stack.extend(node_inputs(&graph.nodes[i]));
        }
        let mut is_root = vec![false; n];
        for &r in &roots {
            is_root[r] = true;
        }
        let fusion = plan_fusion(&graph, &shapes, &live, &is_root);

        let buffer_child = |c: usize, target: &[usize]| ChildRef {
            source: ChildSource::Buffer(c),
            strides: broadcast_strides(&shapes[c], target),
            same_shape: shapes[c] == target,
        };

        let mut groups: Vec<GroupPlan> = Vec::with_capacity(fusion.groups.len());
        for members in &fusion.groups {
            let leader = *members.last().unwrap();
            let out_shape = shapes[leader].clone();
            // slot/temp assignment mirrors execution order.
            let mut slot_of: Vec<Option<usize>> = vec![None; n];
            let mut temp_of: Vec<Option<usize>> = vec![None; n];
            let mut small_members: Vec<MemberPlan> = Vec::new();
            let mut main_members: Vec<MemberPlan> = Vec::new();
            for &m in members {
                let inputs = node_inputs(&graph.nodes[m]);
                if shapes[m] == out_shape {
                    slot_of[m] = Some(main_members.len());
                    let inputs: Vec<ChildRef> = inputs
                        .iter()
                        .map(|&c| {
                            if let Some(slot) = slot_of[c] {
                                ChildRef {
                                    source: ChildSource::Slot(slot),
                                    strides: Vec::new(),
                                    same_shape: true,
                                }
                            } else if let Some(t) = temp_of[c] {
                                ChildRef {
                                    source: ChildSource::Temp(t),
                                    strides: broadcast_strides(&shapes[c], &out_shape),
                                    same_shape: false,
                                }
                            } else {
                                buffer_child(c, &out_shape)
                            }
                        })
                        .collect();
                    main_members.push(MemberPlan {
                        op: Op::of(&graph.nodes[m])?,
                        out_shape: out_shape.clone(),
                        all_same: inputs.iter().all(|c| c.same_shape),
                        inputs,
                    });
                } else {
                    // Small members can only read buffers or earlier temps.
                    temp_of[m] = Some(small_members.len());
                    let target = shapes[m].clone();
                    let inputs: Vec<ChildRef> = inputs
                        .iter()
                        .map(|&c| {
                            if let Some(t) = temp_of[c] {
                                ChildRef {
                                    source: ChildSource::Temp(t),
                                    strides: broadcast_strides(&shapes[c], &target),
                                    same_shape: shapes[c] == target,
                                }
                            } else {
                                buffer_child(c, &target)
                            }
                        })
                        .collect();
                    small_members.push(MemberPlan {
                        op: Op::of(&graph.nodes[m])?,
                        out_shape: target,
                        all_same: inputs.iter().all(|c| c.same_shape),
                        inputs,
                    });
                }
            }
            let all_same = main_members
                .iter()
                .all(|m| m.inputs.iter().all(|c| c.same_shape));
            groups.push(GroupPlan {
                leader,
                out_shape,
                all_same,
                small_members,
                main_members,
            });
        }

        let mut ewise: Vec<Option<MemberPlan>> = (0..n).map(|_| None).collect();
        for (idx, node) in graph.nodes.iter().enumerate() {
            if !live[idx] || fusion.group_of[idx].is_some() || !is_elementwise(node) {
                continue;
            }
            let target = shapes[idx].clone();
            let inputs: Vec<ChildRef> = node_inputs(node)
                .iter()
                .map(|&c| buffer_child(c, &target))
                .collect();
            ewise[idx] = Some(MemberPlan {
                op: Op::of(node)?,
                all_same: inputs.iter().all(|c| c.same_shape),
                inputs,
                out_shape: target,
            });
        }

        // Consumer counts over live edges only. A node read twice by one
        // consumer counts twice, which is what the countdown needs. Used
        // by the candle evaluator, which has no fusion groups, so a
        // node's readers are exactly the nodes listing it as an input.
        let mut consumers = vec![0usize; n];
        for (i, node) in graph.nodes.iter().enumerate() {
            if !live[i] {
                continue;
            }
            for input in node_inputs(node) {
                consumers[input] += 1;
            }
        }
        let target = Target::parse(graph.device.as_deref());

        Ok(PreparedGraph {
            graph,
            shapes,
            roots,
            live,
            consumers,
            is_root,
            group_of: fusion.group_of,
            groups,
            ewise,
            target,
        })
    }
}

static PLAN_CACHE: OnceLock<Mutex<HashMap<String, Arc<PreparedGraph>>>> = OnceLock::new();

/// Parse + prepare, cached on the full JSON (which determines everything).
fn prepared(graph_json: &str) -> Result<Arc<PreparedGraph>> {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = cache.lock().unwrap().get(graph_json) {
        return Ok(p.clone());
    }
    let graph: Graph = serde_json::from_str(graph_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("invalid graph JSON: {e}")))?;
    let prep = Arc::new(PreparedGraph::prepare(graph).map_err(to_napi_err)?);
    let mut map = cache.lock().unwrap();
    // Bounded: compiled functions cache one plan each; a pathological
    // caller building fresh graphs forever just falls back to re-planning.
    if map.len() >= 128 {
        map.clear();
    }
    map.insert(graph_json.to_string(), prep.clone());
    Ok(prep)
}

// Prepared-plan handles. compile() replays one graph thousands of times,
// and going through `prepared()` each time means shipping a JSON string
// across the FFI boundary and hashing all of it just to find the plan
// again — for a rolled-out automaton that string is hundreds of
// kilobytes. prepareGraph() parses and plans once, returns a handle, and
// evalPrepared() takes the handle instead.
static PLAN_HANDLES: OnceLock<Mutex<HashMap<u32, Arc<PreparedGraph>>>> = OnceLock::new();
static NEXT_HANDLE: Mutex<u32> = Mutex::new(1);

fn handles() -> &'static Mutex<HashMap<u32, Arc<PreparedGraph>>> {
    PLAN_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Parse and plan a graph once, returning a handle for `evalPrepared`.
#[napi(js_name = "prepareGraph")]
pub fn prepare_graph(graph_json: String) -> Result<u32> {
    let graph: Graph = serde_json::from_str(&graph_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("invalid graph JSON: {e}")))?;
    let prep = Arc::new(PreparedGraph::prepare(graph).map_err(to_napi_err)?);
    let mut next = NEXT_HANDLE.lock().unwrap();
    let handle = *next;
    *next += 1;
    handles().lock().unwrap().insert(handle, prep);
    Ok(handle)
}

/// Drop a plan created by `prepareGraph`.
#[napi(js_name = "releaseGraph")]
pub fn release_graph(handle: u32) {
    handles().lock().unwrap().remove(&handle);
}

/// Evaluate a plan created by `prepareGraph`.
#[napi(js_name = "evalPrepared")]
pub fn eval_prepared(
    handle: u32,
    leaves: Float32Array,
    seed: u32,
) -> Result<Readback> {
    let prep = handles()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                format!("unknown prepared graph {handle}"),
            )
        })?;
    evaluate(&prep, &leaves, seed)
}

#[inline]
fn read_ref(
    cr: &ChildRef,
    i: usize,
    coords: &[usize],
    buffers: &[Option<Vec<f32>>],
    temps: &[Vec<f32>],
    scratch: &[f32],
) -> f32 {
    match cr.source {
        ChildSource::Slot(slot) => scratch[slot],
        ChildSource::Temp(t) => read_bcast(&temps[t], &cr.strides, cr.same_shape, i, coords),
        ChildSource::Buffer(b) => read_bcast(
            buffers[b].as_deref().unwrap_or(&[]),
            &cr.strides,
            cr.same_shape,
            i,
            coords,
        ),
    }
}

/// `read_ref` for a group where nothing broadcasts: the flat index is the
/// only index there is.
#[inline]
fn read_flat(
    cr: &ChildRef,
    i: usize,
    buffers: &[Option<Vec<f32>>],
    temps: &[Vec<f32>],
    scratch: &[f32],
) -> f32 {
    match cr.source {
        ChildSource::Slot(slot) => scratch[slot],
        ChildSource::Temp(t) => temps[t][i],
        ChildSource::Buffer(b) => buffers[b].as_deref().unwrap_or(&[])[i],
    }
}

/// How many elements one thread takes at a time. Small enough that a big
/// pass spreads over the cores, large enough that rayon's bookkeeping and
/// the per-chunk scratch allocation stay noise.
const CHUNK: usize = 8192;

/// Below this many elements a pass runs on the calling thread: the work is
/// smaller than the cost of handing it out.
const PARALLEL_MIN: usize = 16384;

/// Run `body` over `out` in parallel chunks (or in place if it is small),
/// giving it each chunk together with the flat index the chunk starts at.
fn over_chunks(out: &mut [f32], body: impl Fn(usize, &mut [f32]) + Send + Sync) {
    if out.len() < PARALLEL_MIN {
        body(0, out);
        return;
    }
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, slice)| body(c * CHUNK, slice));
}

/// One elementwise op over its own output shape, into a fresh buffer.
fn exec_member(
    plan: &MemberPlan,
    buffers: &[Option<Vec<f32>>],
    temps: &[Vec<f32>],
) -> Vec<f32> {
    let mut out = vec![0f32; prod(&plan.out_shape)];
    let shape = &plan.out_shape;
    let inputs = &plan.inputs;
    if plan.all_same {
        over_chunks(&mut out, |base, slice| {
            for (k, dst) in slice.iter_mut().enumerate() {
                let i = base + k;
                *dst = match plan.op {
                    Op::Bin(kind, p) => apply_bin(
                        kind,
                        p,
                        read_flat(&inputs[0], i, buffers, temps, &[]),
                        read_flat(&inputs[1], i, buffers, temps, &[]),
                    ),
                    Op::Un(kind, p) => apply_un(
                        kind,
                        p,
                        read_flat(&inputs[0], i, buffers, temps, &[]),
                    ),
                };
            }
        });
        return out;
    }
    over_chunks(&mut out, |base, slice| {
        let mut coords = vec![0usize; shape.len()];
        for (k, dst) in slice.iter_mut().enumerate() {
            let i = base + k;
            flat_to_coords(i, shape, &mut coords);
            *dst = match plan.op {
                Op::Bin(kind, p) => apply_bin(
                    kind,
                    p,
                    read_ref(&inputs[0], i, &coords, buffers, temps, &[]),
                    read_ref(&inputs[1], i, &coords, buffers, temps, &[]),
                ),
                Op::Un(kind, p) => apply_un(
                    kind,
                    p,
                    read_ref(&inputs[0], i, &coords, buffers, temps, &[]),
                ),
            };
        }
    });
    out
}

/// A whole fused group in one pass over the output: every member is
/// evaluated per element into a scratch slot, so the intermediate values
/// of a chain of elementwise ops never reach memory. Only the leader's
/// value is written out.
fn exec_group(plan: &GroupPlan, buffers: &[Option<Vec<f32>>]) -> Vec<f32> {
    // Members smaller than the output shape are evaluated first, into
    // temps: their inputs can only be external buffers or earlier temps.
    let mut temps: Vec<Vec<f32>> = Vec::with_capacity(plan.small_members.len());
    for sm in &plan.small_members {
        temps.push(exec_member(sm, buffers, &temps));
    }
    let members = &plan.main_members;
    let last = members.len() - 1;
    let shape = &plan.out_shape;
    let mut out = vec![0f32; prod(shape)];
    // With no broadcast inputs anywhere in the group, the flat index is
    // the only index anyone needs — worth its own loop, since decomposing
    // an index into coordinates costs more than the arithmetic it feeds.
    if plan.all_same {
        over_chunks(&mut out, |base, slice| {
            let mut scratch = vec![0f32; members.len()];
            for (k, dst) in slice.iter_mut().enumerate() {
                let i = base + k;
                for (slot, mm) in members.iter().enumerate() {
                    scratch[slot] = match mm.op {
                        Op::Bin(kind, p) => apply_bin(
                            kind,
                            p,
                            read_flat(&mm.inputs[0], i, buffers, &temps, &scratch),
                            read_flat(&mm.inputs[1], i, buffers, &temps, &scratch),
                        ),
                        Op::Un(kind, p) => apply_un(
                            kind,
                            p,
                            read_flat(&mm.inputs[0], i, buffers, &temps, &scratch),
                        ),
                    };
                }
                *dst = scratch[last];
            }
        });
        return out;
    }
    over_chunks(&mut out, |base, slice| {
        let mut scratch = vec![0f32; members.len()];
        let mut coords = vec![0usize; shape.len()];
        for (k, dst) in slice.iter_mut().enumerate() {
            let i = base + k;
            flat_to_coords(i, shape, &mut coords);
            for (slot, mm) in members.iter().enumerate() {
                scratch[slot] = match mm.op {
                    Op::Bin(kind, p) => apply_bin(
                        kind,
                        p,
                        read_ref(&mm.inputs[0], i, &coords, buffers, &temps, &scratch),
                        read_ref(&mm.inputs[1], i, &coords, buffers, &temps, &scratch),
                    ),
                    Op::Un(kind, p) => apply_un(
                        kind,
                        p,
                        read_ref(&mm.inputs[0], i, &coords, buffers, &temps, &scratch),
                    ),
                };
            }
            *dst = scratch[last];
        }
    });
    out
}

/// Whole-graph execution from a prepared plan: leaf copies + raw loops,
/// no parsing or planning. Returns all roots concatenated.
fn execute(prep: &PreparedGraph, leaves: &[f32], seed: u32) -> candle_core::Result<Vec<f32>> {
    let graph = &prep.graph;
    let n = graph.nodes.len();
    let mut buffers: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !prep.live[idx] {
            continue;
        }
        if let Some(g) = prep.group_of[idx] {
            if prep.groups[g].leader == idx {
                buffers[idx] = Some(exec_group(&prep.groups[g], &buffers));
            }
            continue;
        }
        let started = if profiling() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let get = |i: usize| -> candle_core::Result<&[f32]> {
            buffers.get(i).and_then(|b| b.as_deref()).ok_or_else(|| {
                candle_core::Error::Msg(format!("node references future index {i}"))
            })
        };
        let out = match node {
            Node::Leaf {
                leaf,
                offset,
                shape,
            } => {
                let n = prod(shape);
                leaves
                    .get(*offset..*offset + n)
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "leaf {leaf} needs {n} f32 values at offset {offset}, have {}",
                            leaves.len()
                        ))
                    })?
                    .to_vec()
            }
            Node::Binary { .. } | Node::Unary { .. } => {
                exec_member(prep.ewise[idx].as_ref().unwrap(), &buffers, &[])
            }
            Node::Matmul { a, b } => {
                cpu_matmul(get(*a)?, &prep.shapes[*a], get(*b)?, &prep.shapes[*b])?
            }
            Node::Reduce {
                kind,
                dim,
                keepdim,
                input,
            } => tiny_reduce(kind, *dim, *keepdim, get(*input)?, &prep.shapes[*input])?,
            Node::ReduceAll { kind, input } => tiny_reduce_all(kind, get(*input)?)?,
            Node::BroadcastTo { input, shape } => {
                tiny_broadcast_to(get(*input)?, &prep.shapes[*input], shape)
            }
            Node::Permute { order, input } => {
                tiny_permute(get(*input)?, &prep.shapes[*input], order)
            }
            // Tiny-path buffers are always contiguous, so a view is free.
            Node::View { input, .. } => get(*input)?.to_vec(),
            Node::Narrow {
                dim,
                start,
                length,
                input,
            } => tiny_narrow(get(*input)?, &prep.shapes[*input], *dim, *start, *length),
            Node::Cat { a, b, dim } => {
                tiny_cat(get(*a)?, &prep.shapes[*a], get(*b)?, &prep.shapes[*b], *dim)
            }
            Node::OneHot { classes, input } => tiny_one_hot(*classes, get(*input)?)?,
            Node::IndexSelect { dim, input, index } => tiny_index_select(
                get(*input)?,
                &prep.shapes[*input],
                get(*index)?,
                *dim,
            )?,
            Node::ScatterAdd {
                dim,
                length,
                input,
                index,
            } => tiny_scatter_add(
                get(*input)?,
                &prep.shapes[*input],
                get(*index)?,
                *dim,
                *length,
            )?,
            Node::Random {
                kind,
                stream,
                shape,
            } => random_data(kind, prod(shape), *stream, seed)?,
        };
        if let Some(started) = started {
            record(
                op_kind(node),
                started.elapsed().as_secs_f64(),
                prod(&prep.shapes[idx]),
            );
        }
        buffers[idx] = Some(out);
    }
    let mut out = Vec::new();
    for &r in &prep.roots {
        out.extend_from_slice(buffers.get(r).and_then(|b| b.as_deref()).ok_or_else(|| {
            candle_core::Error::Msg(format!("root references missing node {r}"))
        })?);
    }
    Ok(out)
}

// Accelerate's BLAS, for the one op where a hand loop cannot compete.
// candle links the same framework; declaring sgemm directly lets the CPU
// evaluator use it without going through a candle tensor.
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

/// Row-major C = A·B for contiguous slices. Rows of A are handed out in
/// blocks so the work spreads over cores whatever BLAS decides to do.
#[cfg(target_os = "macos")]
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if c.is_empty() || k == 0 {
        return;
    }
    let block = ((m + rayon::current_num_threads() - 1)
        / rayon::current_num_threads())
    .max(64);
    let run = |rows: usize, a: &[f32], c: &mut [f32]| unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    };
    if m <= block {
        run(m, a, c);
        return;
    }
    c.par_chunks_mut((block * n).max(1))
        .zip(a.par_chunks((block * k).max(1)))
        .for_each(|(c, a)| run(a.len() / k.max(1), a, c));
}

/// Everywhere without Accelerate: a cache-friendly triple loop.
#[cfg(not(target_os = "macos"))]
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if c.is_empty() || k == 0 {
        return;
    }
    let block = ((m + rayon::current_num_threads() - 1)
        / rayon::current_num_threads())
    .max(64);
    c.par_chunks_mut((block * n).max(1))
        .zip(a.par_chunks((block * k).max(1)))
        .for_each(|(c, a)| {
            for i in 0..a.len() / k.max(1) {
                for l in 0..k {
                    let av = a[i * k + l];
                    for j in 0..n {
                        c[i * n + j] += av * b[l * n + j];
                    }
                }
            }
        });
}

/// Matmul with typenet's batch-dim broadcasting (candle does not do it
/// either), each batch cell going through `gemm`.
fn cpu_matmul(
    adata: &[f32],
    ashape: &[usize],
    bdata: &[f32],
    bshape: &[usize],
) -> candle_core::Result<Vec<f32>> {
    let (ar, br) = (ashape.len(), bshape.len());
    let (m, k, n) = (ashape[ar - 2], ashape[ar - 1], bshape[br - 1]);
    let batch = broadcast_dim_vecs(&ashape[..ar - 2], &bshape[..br - 2])?;
    let a_bs = broadcast_strides(&ashape[..ar - 2], &batch);
    let b_bs = broadcast_strides(&bshape[..br - 2], &batch);
    let nb = prod(&batch);
    let mut out = vec![0f32; nb * m * n];
    let mut bcoords = vec![0usize; batch.len()];
    for bi in 0..nb {
        flat_to_coords(bi, &batch, &mut bcoords);
        let (mut ao, mut bo) = (0usize, 0usize);
        for j in 0..batch.len() {
            ao += bcoords[j] * a_bs[j];
            bo += bcoords[j] * b_bs[j];
        }
        let (ao, bo) = (ao * m * k, bo * k * n);
        let oo = bi * m * n;
        gemm(
            &adata[ao..ao + m * k],
            &bdata[bo..bo + k * n],
            &mut out[oo..oo + m * n],
            m,
            k,
            n,
        );
    }
    Ok(out)
}

/// Read an index buffer of integral f32s, bounds-checked against `rows`.
fn read_indices(index: &[f32], rows: usize, what: &str) -> candle_core::Result<Vec<usize>> {
    index
        .iter()
        .map(|&v| {
            if v.fract() != 0.0 || v < 0.0 || v as usize >= rows {
                return Err(candle_core::Error::Msg(format!(
                    "{what}: index {v} out of range for {rows} rows"
                )));
            }
            Ok(v as usize)
        })
        .collect()
}

fn tiny_index_select(
    data: &[f32],
    shape: &[usize],
    index: &[f32],
    dim: usize,
) -> candle_core::Result<Vec<f32>> {
    let rows = shape[dim];
    let indices = read_indices(index, rows, "indexSelect")?;
    let inner = row_major_strides(shape)[dim];
    let outer = prod(&shape[..dim]);
    let mut out = vec![0f32; outer * indices.len() * inner];
    let picked = indices.len();
    // Output rows are independent, so hand them out in blocks.
    out.par_chunks_mut(inner.max(1) * 64)
        .enumerate()
        .for_each(|(c, slice)| {
            let start = c * 64;
            for (r, dst) in slice.chunks_mut(inner.max(1)).enumerate() {
                let flat = start + r;
                let base = (flat / picked * rows + indices[flat % picked]) * inner;
                dst.copy_from_slice(&data[base..base + inner]);
            }
        });
    Ok(out)
}

fn tiny_scatter_add(
    data: &[f32],
    shape: &[usize],
    index: &[f32],
    dim: usize,
    length: usize,
) -> candle_core::Result<Vec<f32>> {
    let indices = read_indices(index, length, "scatterAdd")?;
    let inner = row_major_strides(shape)[dim].max(1);
    let outer = prod(&shape[..dim]);
    let src_rows = shape[dim];
    let mut out = vec![0f32; outer * length * inner];
    let slice = length * inner;

    // Colliding indices make this the one op whose writes cannot simply be
    // split by output range. But slices along the dims *outside* `dim` are
    // fully independent, so when there is more than one of them they are
    // the natural unit of work.
    if outer > 1 {
        out.par_chunks_mut(slice.max(1)).enumerate().for_each(|(i, out)| {
            for (j, &row) in indices.iter().enumerate() {
                let to = row * inner;
                let from = (i * src_rows + j) * inner;
                for k in 0..inner {
                    out[to + k] += data[from + k];
                }
            }
        });
        return Ok(out);
    }

    // A single slice — the usual dim-0 aggregation over an edge list. Give
    // each thread a block of output rows and let it scan the index for the
    // entries landing in its block: an index entry is 8 bytes against the
    // `inner` floats a hit copies, so re-reading it per thread beats
    // coordinating the writes.
    let per = (length / rayon::current_num_threads().max(1)).max(1);
    out.par_chunks_mut(per * inner)
        .enumerate()
        .for_each(|(c, out)| {
            let lo = c * per;
            let hi = lo + out.len() / inner;
            for (j, &row) in indices.iter().enumerate() {
                if row < lo || row >= hi {
                    continue;
                }
                let to = (row - lo) * inner;
                let from = j * inner;
                for k in 0..inner {
                    out[to + k] += data[from + k];
                }
            }
        });
    Ok(out)
}

fn tiny_reduce(
    kind: &str,
    dim: usize,
    keepdim: bool,
    data: &[f32],
    shape: &[usize],
) -> candle_core::Result<Vec<f32>> {
    let strides = row_major_strides(shape);
    let d = shape[dim];
    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    let n_out = prod(&out_shape);
    let step = strides[dim];
    let rank = shape.len();
    let kind = Reduce::parse(kind)?;
    let mut out = vec![0f32; n_out];
    over_chunks(&mut out, |base_i, slice| {
        let mut coords = vec![0usize; rank];
        for (k, dst) in slice.iter_mut().enumerate() {
            // Output element i maps to the input coords with the reduced
            // coordinate pinned at 0; walking `step` from there sweeps it.
            let mut rem = base_i + k;
            for j in (0..rank).rev() {
                let size = if j == dim { 1 } else { shape[j] };
                coords[j] = rem % size;
                rem /= size;
            }
            let mut base = 0usize;
            for j in 0..rank {
                base += coords[j] * strides[j];
            }
            *dst = match kind {
                Reduce::Sum => {
                    let mut acc = 0f32;
                    for dd in 0..d {
                        acc += data[base + dd * step];
                    }
                    acc
                }
                Reduce::Max => {
                    let mut acc = data[base];
                    for dd in 1..d {
                        let v = data[base + dd * step];
                        if v > acc {
                            acc = v;
                        }
                    }
                    acc
                }
                // First index wins on ties, matching the eager kernel.
                Reduce::Argmax => {
                    let mut best = 0usize;
                    let mut acc = data[base];
                    for dd in 1..d {
                        let v = data[base + dd * step];
                        if v > acc {
                            acc = v;
                            best = dd;
                        }
                    }
                    best as f32
                }
            };
        }
    });
    Ok(out)
}

#[derive(Clone, Copy)]
enum Reduce {
    Sum,
    Max,
    Argmax,
}

impl Reduce {
    fn parse(kind: &str) -> candle_core::Result<Self> {
        Ok(match kind {
            "sum" => Reduce::Sum,
            "max" => Reduce::Max,
            "argmax" => Reduce::Argmax,
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown reduce op: {other}"
                )))
            }
        })
    }
}

fn tiny_reduce_all(kind: &str, data: &[f32]) -> candle_core::Result<Vec<f32>> {
    match kind {
        "sum" => Ok(vec![data.iter().sum()]),
        "max" => Ok(vec![data.iter().copied().fold(f32::NEG_INFINITY, |a, b| {
            if b > a {
                b
            } else {
                a
            }
        })]),
        other => Err(candle_core::Error::Msg(format!(
            "unknown reduceAll op: {other}"
        ))),
    }
}

fn tiny_broadcast_to(data: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    let strides = broadcast_strides(from, to);
    let same = from == to;
    let mut out = vec![0f32; prod(to)];
    over_chunks(&mut out, |base, slice| {
        let mut coords = vec![0usize; to.len()];
        for (k, dst) in slice.iter_mut().enumerate() {
            let i = base + k;
            flat_to_coords(i, to, &mut coords);
            *dst = read_bcast(data, &strides, same, i, &coords);
        }
    });
    out
}

fn tiny_permute(data: &[f32], shape: &[usize], order: &[usize]) -> Vec<f32> {
    let in_strides = row_major_strides(shape);
    let out_shape: Vec<usize> = order.iter().map(|&d| shape[d]).collect();
    // Output coords -> input index via the permuted input strides.
    let strides: Vec<usize> = order.iter().map(|&d| in_strides[d]).collect();
    let mut out = vec![0f32; prod(&out_shape)];
    over_chunks(&mut out, |base, slice| {
        let mut coords = vec![0usize; out_shape.len()];
        for (k, dst) in slice.iter_mut().enumerate() {
            let i = base + k;
            flat_to_coords(i, &out_shape, &mut coords);
            let mut idx = 0usize;
            for j in 0..coords.len() {
                idx += coords[j] * strides[j];
            }
            *dst = data[idx];
        }
    });
    out
}

/// A window along one dim is a run of contiguous blocks, so this is a
/// series of copies rather than a per-element index computation.
fn tiny_narrow(data: &[f32], shape: &[usize], dim: usize, start: usize, length: usize) -> Vec<f32> {
    let inner = row_major_strides(shape)[dim];
    let outer = prod(&shape[..dim]);
    let rows = shape[dim];
    let block = length * inner;
    let mut out = vec![0f32; outer * block];
    let copy = |i: usize, out: &mut [f32]| {
        let from = (i * rows + start) * inner;
        out.copy_from_slice(&data[from..from + block]);
    };
    if out.len() < PARALLEL_MIN {
        out.chunks_mut(block.max(1))
            .enumerate()
            .for_each(|(i, out)| copy(i, out));
    } else {
        out.par_chunks_mut(block.max(1))
            .enumerate()
            .for_each(|(i, out)| copy(i, out));
    }
    out
}

/// Likewise a concatenation: each output slice along `dim` is one block
/// from each side, copied whole.
fn tiny_cat(
    adata: &[f32],
    ashape: &[usize],
    bdata: &[f32],
    bshape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let inner = row_major_strides(ashape)[dim];
    let outer = prod(&ashape[..dim]);
    let (arows, brows) = (ashape[dim], bshape[dim]);
    let (ablock, bblock) = (arows * inner, brows * inner);
    let mut out = vec![0f32; outer * (ablock + bblock)];
    let copy = |i: usize, out: &mut [f32]| {
        out[..ablock].copy_from_slice(&adata[i * ablock..(i + 1) * ablock]);
        out[ablock..].copy_from_slice(&bdata[i * bblock..(i + 1) * bblock]);
    };
    let block = (ablock + bblock).max(1);
    if out.len() < PARALLEL_MIN {
        out.chunks_mut(block).enumerate().for_each(|(i, out)| copy(i, out));
    } else {
        out.par_chunks_mut(block)
            .enumerate()
            .for_each(|(i, out)| copy(i, out));
    }
    out
}

fn tiny_one_hot(classes: usize, data: &[f32]) -> candle_core::Result<Vec<f32>> {
    let mut out = vec![0f32; data.len() * classes];
    for (i, &v) in data.iter().enumerate() {
        if v.fract() != 0.0 || v < 0.0 || v >= classes as f32 {
            return Err(candle_core::Error::Msg(format!(
                "oneHot: target {v} out of range for {classes} classes"
            )));
        }
        out[i * classes + v as usize] = 1.0;
    }
    Ok(out)
}

/// Whole-graph evaluation on Vec<f32> buffers is plan/exec based — see
/// PreparedGraph above. This stub remains only to keep the section
/// structure obvious: eval_graph calls prepared() + execute().


// Candle's dim-reductions (sum/max/min) squeeze the dim; typenet's
// keepdim semantics require re-inserting it.
fn reinsert_dim(t: Tensor, dim: usize, keepdim: bool) -> candle_core::Result<Tensor> {
    if keepdim { t.unsqueeze(dim) } else { Ok(t) }
}

fn eval_reduce(
    kind: &str,
    dim: usize,
    keepdim: bool,
    a: &Tensor,
) -> candle_core::Result<Tensor> {
    match kind {
        "sum" => reinsert_dim(a.sum(dim)?, dim, keepdim),
        "max" => reinsert_dim(a.max(dim)?, dim, keepdim),
        "argmax" => {
            // First index of the max along `dim`: mask ties, take the min
            // of (index or +inf) so the earliest index wins.
            let input = a.contiguous()?;
            let mut keep_shape = input.dims().to_vec();
            keep_shape[dim] = 1;
            let best = input
                .max(dim)?
                .reshape(keep_shape)?
                .broadcast_as(input.shape())?
                .contiguous()?;
            let mask = input.eq(&best)?;
            let n = input.dim(dim)?;
            let idx = Tensor::arange(0u32, n as u32, input.device())?;
            let mut idx_shape = vec![1usize; input.rank()];
            idx_shape[dim] = n;
            let idx = idx
                .reshape(idx_shape)?
                .broadcast_as(input.shape())?
                .contiguous()?
                .to_dtype(DType::F32)?;
            let big = (input.ones_like()? * f64::from(u32::MAX))?;
            let masked = mask.where_cond(&idx, &big)?;
            reinsert_dim(masked.min(dim)?, dim, keepdim)
        }
        other => Err(candle_core::Error::Msg(format!(
            "unknown reduce op: {other}"
        ))),
    }
}

fn eval_one_hot(classes: usize, a: &Tensor) -> candle_core::Result<Tensor> {
    let flat = a.contiguous()?.flatten_all()?;
    let values = flat.to_vec1::<f32>()?;
    for &v in &values {
        if v.fract() != 0.0 || v < 0.0 || v >= classes as f32 {
            return Err(candle_core::Error::Msg(format!(
                "oneHot: target {v} out of range for {classes} classes"
            )));
        }
    }
    let n = values.len();
    let targets = flat
        .to_dtype(DType::U32)?
        .reshape((n, 1))?
        .broadcast_as((n, classes))?
        .contiguous()?;
    let range = Tensor::arange(0u32, classes as u32, a.device())?
        .reshape((1, classes))?
        .broadcast_as((n, classes))?
        .contiguous()?;
    targets.eq(&range)?.to_dtype(DType::F32)
}

/// typenet has no integer dtype, so index tensors arrive as f32 holding
/// integral values. candle's gather/scatter kernels want U32.
fn index_u32(index: &Tensor) -> candle_core::Result<Tensor> {
    index.contiguous()?.flatten_all()?.to_dtype(DType::U32)
}

/// Label a node by op kind for the profile table; binary and unary nodes
/// report their specific kind, which is where the interesting differences
/// between them show up.
fn op_kind(node: &Node) -> &str {
    match node {
        Node::Leaf { .. } => "leaf",
        Node::Binary { kind, .. } => kind,
        Node::Unary { kind, .. } => kind,
        Node::Matmul { .. } => "matmul",
        Node::Reduce { .. } => "reduce",
        Node::ReduceAll { .. } => "reduceAll",
        Node::BroadcastTo { .. } => "broadcastTo",
        Node::Permute { .. } => "permute",
        Node::View { .. } => "view",
        Node::Narrow { .. } => "narrow",
        Node::Cat { .. } => "cat",
        Node::OneHot { .. } => "oneHot",
        Node::IndexSelect { .. } => "indexSelect",
        Node::ScatterAdd { .. } => "scatterAdd",
        Node::Random { kind, .. } => kind,
    }
}

/// The value of a node that is a one-element leaf, read straight out of
/// the leaf buffer. Constants coerced from JS numbers land here, and
/// reading them this way costs nothing — no device readback, since the
/// leaf buffer is host memory that has not been uploaded yet.
fn scalar_leaf(graph: &Graph, leaves: &[f32], at: usize) -> Option<f32> {
    match &graph.nodes[at] {
        Node::Leaf { offset, shape, .. } if prod(shape) == 1 => {
            leaves.get(*offset).copied()
        }
        _ => None,
    }
}

/// The u32 form of an index node, converted on first use and kept.
fn cached_index(
    cache: &mut [Option<Tensor>],
    at: usize,
    index: &Tensor,
) -> candle_core::Result<Tensor> {
    if cache[at].is_none() {
        cache[at] = Some(index_u32(index)?);
    }
    Ok(cache[at].as_ref().unwrap().clone())
}

fn run_graph(
    prep: &PreparedGraph,
    leaves: &[f32],
    device: &Device,
    seed: u32,
) -> candle_core::Result<Vec<Tensor>> {
    let graph = &prep.graph;
    let n = graph.nodes.len();
    // Liveness (dead nodes are never touched — a dead leaf is not even
    // copied to the device) and consumer counts come from the plan.
    let mut remaining = prep.consumers.clone();
    let mut outputs: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    // Index tensors are read several times per rolled-out step — once per
    // gather or scatter — and each read would otherwise re-run the f32 to
    // u32 cast. Convert once and keep it for as long as the f32 original
    // is alive.
    let mut indices: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !prep.live[idx] {
            continue;
        }
        let get = |i: usize| -> candle_core::Result<&Tensor> {
            outputs.get(i).and_then(|t| t.as_ref()).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "node {i} was already released or never computed"
                ))
            })
        };
        let started = if profiling() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let out = match node {
            Node::Leaf {
                leaf,
                offset,
                shape,
            } => {
                let n = prod(shape);
                let data = leaves.get(*offset..*offset + n).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "leaf {leaf} needs {n} f32 values at offset {offset}, have {}",
                        leaves.len()
                    ))
                })?;
                Tensor::from_vec(data.to_vec(), shape.clone(), device)?
            }
            Node::Binary {
                kind,
                parameter,
                a,
                b,
                ..
            } => {
                // Scaling or shifting by a constant is a one-element
                // operand, and candle's broadcast path walks a general
                // strided index for it — measured 6x slower per element
                // than the same op between equal shapes. `affine`
                // (x·mul + add) is one fused kernel, and for these three
                // ops the rewrite is exact, not an approximation.
                let sa = scalar_leaf(graph, leaves, *a);
                let sb = scalar_leaf(graph, leaves, *b);
                match (kind.as_str(), sa, sb) {
                    ("mul", _, Some(s)) => get(*a)?.affine(s as f64, 0.0)?,
                    ("mul", Some(s), _) => get(*b)?.affine(s as f64, 0.0)?,
                    ("add", _, Some(s)) => get(*a)?.affine(1.0, s as f64)?,
                    ("add", Some(s), _) => get(*b)?.affine(1.0, s as f64)?,
                    ("sub", _, Some(s)) => {
                        get(*a)?.affine(1.0, -(s as f64))?
                    }
                    ("sub", Some(s), _) => {
                        get(*b)?.affine(-1.0, s as f64)?
                    }
                    _ => eval_binary(kind, *parameter, get(*a)?, get(*b)?)?,
                }
            }
            Node::Unary {
                kind,
                parameter,
                input,
            } => eval_unary(kind, *parameter, get(*input)?)?,
            Node::Matmul { a, b } => {
                let a = get(*a)?;
                let b = get(*b)?;
                let ar = a.rank();
                let br = b.rank();
                let m = a.dim(ar - 2)?;
                let k = a.dim(ar - 1)?;
                let n = b.dim(br - 1)?;
                // typenet broadcasts batch dims; candle does not.
                let batch = candle_core::Shape::from_dims(&a.dims()[..ar - 2])
                    .broadcast_shape_binary_op(
                        &candle_core::Shape::from_dims(&b.dims()[..br - 2]),
                        "matmul",
                    )?;
                let mut a_shape = batch.dims().to_vec();
                a_shape.extend([m, k]);
                let mut b_shape = batch.dims().to_vec();
                b_shape.extend([k, n]);
                // Only materialize an operand when its batch dims actually
                // need broadcasting. candle's matmul reads strides itself,
                // and every backward pass transposes one operand — making
                // those contiguous first meant copying both matrices on
                // the way into every gradient matmul.
                let owned_a;
                let a = if a.dims() == a_shape.as_slice() {
                    a
                } else {
                    owned_a = a.broadcast_as(a_shape)?.contiguous()?;
                    &owned_a
                };
                let owned_b;
                let b = if b.dims() == b_shape.as_slice() {
                    b
                } else {
                    owned_b = b.broadcast_as(b_shape)?.contiguous()?;
                    &owned_b
                };
                a.matmul(b)?
            }
            Node::Reduce {
                kind,
                dim,
                keepdim,
                input,
            } => eval_reduce(kind, *dim, *keepdim, get(*input)?)?,
            Node::ReduceAll { kind, input } => {
                let flat = get(*input)?.contiguous()?.flatten_all()?;
                let out = match kind.as_str() {
                    "sum" => flat.sum(0)?,
                    "max" => flat.max(0)?,
                    other => {
                        return Err(candle_core::Error::Msg(format!(
                            "unknown reduceAll op: {other}"
                        )))
                    }
                };
                out.reshape(())?
            }
            Node::BroadcastTo { input, shape } => {
                get(*input)?.broadcast_as(shape.clone())?.contiguous()?
            }
            Node::Permute { order, input } => get(*input)?.permute(order.clone())?,
            Node::View { input, shape } => {
                get(*input)?.contiguous()?.reshape(shape.clone())?
            }
            Node::Narrow {
                dim,
                start,
                length,
                input,
            } => get(*input)?.narrow(*dim, *start, *length)?.contiguous()?,
            Node::Cat { a, b, dim } => {
                let a = get(*a)?.contiguous()?;
                let b = get(*b)?.contiguous()?;
                Tensor::cat(&[&a, &b], *dim)?
            }
            Node::OneHot { classes, input } => eval_one_hot(*classes, get(*input)?)?,
            Node::IndexSelect { dim, input, index } => {
                let keys = cached_index(&mut indices, *index, get(*index)?)?;
                get(*input)?.contiguous()?.index_select(&keys, *dim)?
            }
            Node::ScatterAdd {
                dim,
                length,
                input,
                index,
            } => {
                let keys = cached_index(&mut indices, *index, get(*index)?)?;
                let src = get(*input)?.contiguous()?;
                let mut shape = src.dims().to_vec();
                shape[*dim] = *length;
                Tensor::zeros(shape, DType::F32, device)?
                    .index_add(&keys, &src, *dim)?
            }
            Node::Random {
                kind,
                stream,
                shape,
            } => Tensor::from_vec(
                random_data(kind, prod(shape), *stream, seed)?,
                shape.clone(),
                device,
            )?,
        };
        if let Some(started) = started {
            record(
                op_kind(node),
                started.elapsed().as_secs_f64(),
                prod(&prep.shapes[idx]),
            );
        }
        outputs[idx] = Some(out);
        // Release every input nothing else will read. On a rolled-out
        // automaton this is the difference between holding one activation
        // per graph node and holding only what the backward pass still
        // needs, which is also what PyTorch holds.
        for input in node_inputs(node) {
            remaining[input] -= 1;
            if remaining[input] == 0 && !prep.is_root[input] {
                outputs[input] = None;
                indices[input] = None;
            }
        }
    }
    prep.roots
        .iter()
        .map(|&i| {
            outputs.get(i).and_then(|t| t.clone()).ok_or_else(|| {
                candle_core::Error::Msg(format!("root references missing node {i}"))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Zero-copy readback: hand the f32 Vec to JS as an external ArrayBuffer
// (modeled on effect-torch packages/native/src/lib.rs:36-104).
// ---------------------------------------------------------------------------

struct FinalizeHint {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

unsafe extern "C" fn finalize_readback(
    _env: napi::sys::napi_env,
    _data: *mut std::ffi::c_void,
    hint: *mut std::ffi::c_void,
) {
    let hint = unsafe { Box::from_raw(hint as *mut FinalizeHint) };
    drop(unsafe { Vec::from_raw_parts(hint.ptr, hint.len, hint.cap) });
}

pub struct Readback {
    data: *mut u8,
    byte_len: usize,
    hint: FinalizeHint,
}

unsafe impl Send for Readback {}

impl ToNapiValue for Readback {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        value: Self,
    ) -> Result<napi::sys::napi_value> {
        let hint = Box::into_raw(Box::new(value.hint)) as *mut std::ffi::c_void;
        let mut result = std::ptr::null_mut();
        napi::check_status!(
            unsafe {
                napi::sys::napi_create_external_arraybuffer(
                    env,
                    value.data as *mut std::ffi::c_void,
                    value.byte_len,
                    Some(finalize_readback),
                    hint,
                    &mut result,
                )
            },
            "failed to create external arraybuffer"
        )?;
        Ok(result)
    }
}

fn vec_readback(mut vec: Vec<f32>) -> Readback {
    let ptr = vec.as_mut_ptr() as *mut u8;
    let byte_len = vec.len() * std::mem::size_of::<f32>();
    let byte_cap = vec.capacity() * std::mem::size_of::<f32>();
    let f32_len = vec.len();
    std::mem::forget(vec);
    Readback {
        data: ptr,
        byte_len,
        hint: FinalizeHint {
            ptr,
            len: f32_len,
            cap: byte_cap / std::mem::size_of::<f32>(),
        },
    }
}

#[napi(js_name = "evalGraph")]
pub fn eval_graph(graph_json: String, leaves: Float32Array, seed: u32) -> Result<Readback> {
    let prep = prepared(&graph_json)?;
    evaluate(&prep, &leaves, seed)
}

// ---------------------------------------------------------------------------
// Op-kind profiling, off unless TYPENET_PROFILE is set. Wall time and
// element counts per op kind, accumulated across evaluations and drained
// by takeProfile() — enough to tell a bandwidth problem from a dispatch
// problem without a sampling profiler.
// ---------------------------------------------------------------------------

static PROFILE: Mutex<Option<Vec<(String, f64, u64, u64)>>> = Mutex::new(None);

fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TYPENET_PROFILE").is_ok())
}

fn record(kind: &str, seconds: f64, elements: usize) {
    let mut guard = PROFILE.lock().unwrap();
    let rows = guard.get_or_insert_with(Vec::new);
    match rows.iter_mut().find(|(name, ..)| name == kind) {
        Some(row) => {
            row.1 += seconds;
            row.2 += elements as u64;
            row.3 += 1;
        }
        None => rows.push((kind.to_string(), seconds, elements as u64, 1)),
    }
}

/// Op-kind timings gathered since the last call, as a text table.
#[napi(js_name = "takeProfile")]
pub fn take_profile() -> String {
    let mut guard = PROFILE.lock().unwrap();
    let Some(mut rows) = guard.take() else {
        return String::new();
    };
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f64 = rows.iter().map(|r| r.1).sum();
    let mut out = format!(
        "{:<16}{:>10}{:>8}{:>12}{:>12}\n",
        "op", "ms", "share", "calls", "M elem/s"
    );
    for (kind, seconds, elements, calls) in &rows {
        out += &format!(
            "{:<16}{:>10.1}{:>7.1}%{:>12}{:>12.0}\n",
            kind,
            seconds * 1000.0,
            100.0 * seconds / total.max(1e-12),
            calls,
            *elements as f64 / seconds.max(1e-12) / 1e6
        );
    }
    out += &format!("{:<16}{:>10.1}\n", "total", total * 1000.0);
    out
}

/// Overrides the JS side's evaluator choice, for measuring one against
/// another: TYPENET_EVALUATOR=loops | cpu | gpu.
fn forced_target() -> Option<Target> {
    static CHOICE: OnceLock<Option<Target>> = OnceLock::new();
    *CHOICE.get_or_init(|| match std::env::var("TYPENET_EVALUATOR") {
        Ok(name) if !name.is_empty() => Some(Target::parse(Some(name.as_str()))),
        _ => None,
    })
}

/// Run a prepared graph on the evaluator it was planned for.
fn evaluate(prep: &PreparedGraph, leaves: &[f32], seed: u32) -> Result<Readback> {
    let target = forced_target().unwrap_or(prep.target);
    if target == Target::Loops {
        let data = execute(prep, leaves, seed).map_err(to_napi_err)?;
        return Ok(vec_readback(data));
    }
    let device = if target == Target::Accelerator {
        device()
    } else {
        &Device::Cpu
    };
    let outputs = run_graph(prep, leaves, device, seed).map_err(to_napi_err)?;
    // All roots are read back as one concatenated f32 buffer; the JS
    // side slices it per root using the shapes it already knows.
    device.synchronize().map_err(to_napi_err)?;
    let mut flats: Vec<Tensor> = Vec::with_capacity(outputs.len());
    for output in &outputs {
        flats.push(
            output
                .contiguous()
                .and_then(|t| t.flatten_all())
                .map_err(to_napi_err)?,
        );
    }
    // One cat + one readback, no matter how many roots — per-tensor
    // readbacks on Metal each cost a sync.
    let data = if flats.len() == 1 {
        flats.into_iter().next().unwrap()
    } else {
        Tensor::cat(&flats.iter().collect::<Vec<_>>(), 0).map_err(to_napi_err)?
    }
    .to_vec1::<f32>()
    .map_err(to_napi_err)?;
    Ok(vec_readback(data))
}
