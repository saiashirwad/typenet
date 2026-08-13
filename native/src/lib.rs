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
// indices; `roots` lists the output nodes (defaulting to the last node
// for single-root graphs). Leaves index into the `leaves` Float32Array
// as contiguous slices of prod(shape) f32 values.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum Node {
    Leaf {
        leaf: usize,
        offset: usize,
        shape: Vec<usize>,
        /// "float32" (default) | "int32" | "int64". Integer leaves are
        /// only legal as gather/scatter indices; the JS side enforces
        /// that and packs every leaf as its native bytes.
        #[serde(default)]
        dtype: Option<String>,
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
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    Matmul {
        a: usize,
        b: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    Reduce {
        kind: String,
        dim: usize,
        keepdim: bool,
        input: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    ReduceAll {
        kind: String,
        input: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    BroadcastTo {
        input: usize,
        shape: Vec<usize>,
    },
    Permute {
        order: Vec<usize>,
        input: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
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
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    Cat {
        a: usize,
        b: usize,
        dim: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    OneHot {
        classes: usize,
        input: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    /// Gather rows: out[j] = input[index[j]] along `dim`.
    IndexSelect {
        dim: usize,
        input: usize,
        index: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    /// Scatter-add rows into a zero tensor of `length` rows along `dim`:
    /// out[index[j]] += input[j].
    ScatterAdd {
        dim: usize,
        length: usize,
        input: usize,
        index: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
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
/// src/lazy.ts) because it knows the graph's total size before
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
    fn parse(hint: Option<&str>) -> candle_core::Result<Self> {
        match hint {
            None | Some("cpu") => Ok(Target::Cpu),
            Some("loops") => Ok(Target::Loops),
            Some("gpu") => Ok(Target::Accelerator),
            Some(other) => Err(candle_core::Error::Msg(format!(
                "unknown evaluator {other} (expected loops | cpu | gpu)"
            ))),
        }
    }
}

fn prod(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The storage type of a leaf. Compute leaves are `F32`; integer leaves
/// (`I32` / `I64`) feed gather/scatter indices and are read as their
/// native width, so no f32 mantissa limit applies to them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LeafTy {
    F32,
    I32,
    I64,
}

impl LeafTy {
    fn parse(dtype: Option<&str>) -> candle_core::Result<Self> {
        match dtype.unwrap_or("float32") {
            "float32" => Ok(LeafTy::F32),
            "int32" => Ok(LeafTy::I32),
            "int64" => Ok(LeafTy::I64),
            other => Err(candle_core::Error::Msg(format!(
                "unsupported leaf dtype {other}"
            ))),
        }
    }

    fn size(self) -> usize {
        match self {
            LeafTy::F32 | LeafTy::I32 => 4,
            LeafTy::I64 => 8,
        }
    }
}

/// The byte slice of one leaf, bounds-checked against the buffer.
fn leaf_bytes<'a>(
    leaves: &'a [u8],
    leaf: usize,
    offset: usize,
    n: usize,
    width: usize,
) -> candle_core::Result<&'a [u8]> {
    let len = n.checked_mul(width).ok_or_else(|| {
        candle_core::Error::Msg(format!("leaf {leaf} byte size overflows"))
    })?;
    let end = offset.checked_add(len).ok_or_else(|| {
        candle_core::Error::Msg(format!("leaf {leaf} byte range overflows"))
    })?;
    leaves.get(offset..end).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "leaf {leaf} needs {len} bytes at offset {offset}, have {}",
            leaves.len()
        ))
    })
}

/// Read a leaf as f32. Integer leaves are converted exactly — the loop
/// evaluator only runs graphs below the element cap, so their row
/// indices are far below f32's 16.7M exact-integer limit.
fn read_leaf_f32(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
    ty: LeafTy,
) -> candle_core::Result<Vec<f32>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, ty.size())?;
    let mut out = Vec::with_capacity(n);
    match ty {
        LeafTy::F32 => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        LeafTy::I32 => {
            for c in bytes.chunks_exact(4) {
                out.push(i32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        LeafTy::I64 => {
            for c in bytes.chunks_exact(8) {
                let b: [u8; 8] = c.try_into().unwrap();
                out.push(i64::from_ne_bytes(b) as f32);
            }
        }
    }
    Ok(out)
}

fn read_leaf_i32(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
) -> candle_core::Result<Vec<i32>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_leaf_i64(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
) -> candle_core::Result<Vec<i64>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, 8)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| {
            let b: [u8; 8] = c.try_into().unwrap();
            i64::from_ne_bytes(b)
        })
        .collect())
}

/// Indices of the nodes a node directly reads.
fn node_inputs(node: &Node) -> Vec<usize> {
    match node {
        Node::Leaf { .. } => vec![],
        Node::Binary { a, b, .. } => vec![*a, *b],
        Node::Unary { input, .. } => vec![*input],
        Node::Matmul { a, b, .. } => vec![*a, *b],
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
/// The shape the JS side serialized for a node, when it sent one.
fn sent_shape(node: &Node) -> Option<&Vec<usize>> {
    match node {
        Node::Leaf { shape, .. } => Some(shape),
        Node::Binary { shape, .. } => Some(shape),
        Node::BroadcastTo { shape, .. } => Some(shape),
        Node::View { shape, .. } => Some(shape),
        Node::Random { shape, .. } => Some(shape),
        Node::Unary { shape, .. } => shape.as_ref(),
        Node::Matmul { shape, .. } => shape.as_ref(),
        Node::Reduce { shape, .. } => shape.as_ref(),
        Node::ReduceAll { shape, .. } => shape.as_ref(),
        Node::Permute { shape, .. } => shape.as_ref(),
        Node::Narrow { shape, .. } => shape.as_ref(),
        Node::Cat { shape, .. } => shape.as_ref(),
        Node::OneHot { shape, .. } => shape.as_ref(),
        Node::IndexSelect { shape, .. } => shape.as_ref(),
        Node::ScatterAdd { shape, .. } => shape.as_ref(),
    }
}

/// Recomputed shapes are compared against the JS-sent ones in debug
/// builds and under TYPENET_CHECK_SHAPES=1; release trusts JS.
fn shape_check_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        cfg!(debug_assertions)
            || std::env::var("TYPENET_CHECK_SHAPES")
                .map(|v| v == "1")
                .unwrap_or(false)
    })
}

fn node_shapes(graph: &Graph) -> candle_core::Result<Vec<Vec<usize>>> {
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        let shape = match node {
            Node::Leaf { shape, .. } => shape.clone(),
            Node::Binary { shape, .. } => shape.clone(),
            Node::Unary { input, .. } => shapes[*input].clone(),
            Node::Matmul { a, b, .. } => {
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
            Node::Permute { order, input, .. } => {
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
            Node::Cat { a, b, dim, .. } => {
                let mut s = shapes[*a].clone();
                s[*dim] += shapes[*b][*dim];
                s
            }
            Node::OneHot { classes, input, .. } => vec![prod(&shapes[*input]), *classes],
            Node::IndexSelect { dim, input, index, .. } => {
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
        if shape_check_enabled() {
            if let Some(sent) = sent_shape(node) {
                if sent != &shape {
                    return Err(candle_core::Error::Msg(format!(
                        "TYPENET_CHECK_SHAPES: node {i} ({}) recomputed as {shape:?} but JS sent {sent:?}",
                        op_kind(node)
                    )));
                }
            }
        }
        shapes.push(shape);
    }
    Ok(shapes)
}

// ---------------------------------------------------------------------------
// Counter-based random numbers. Element `i` of stream `s` under seed `k`
// is a pure hash of (k, s, i): no state to thread through the evaluator,
// every element independent, and the same arithmetic as the TS side
// (hash32 / unitFloat in src/kernels.ts). Uniform draws therefore match
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
// Tiny-graph CPU evaluator. Graphs the JS side pins with `device: "loops"`
// (≤ LOOP_EVALUATOR_MAX_WORK total elements) are dominated by candle's per-op
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
) -> (FusionPlan, Vec<usize>) {
    let n = graph.nodes.len();
    // Consumer counts over live edges only. A node read twice by one
    // consumer counts twice, which is what the countdown needs. Used by
    // fusion (single-consumer chain rule) and by the candle evaluator.
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
    (FusionPlan { group_of, groups }, consumers)
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
    /// Global node indices this plan reads; ChildSource::Buffer holds an
    /// index into this list (localized after prepare), so execution can
    /// pack just these inputs instead of a whole-graph table.
    buffer_inputs: Vec<usize>,
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
    /// Per-node plans for standalone elementwise nodes, with the global
    /// node indices their localized Buffer sources refer to.
    ewise: Vec<Option<(MemberPlan, Vec<usize>)>>,
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
        let (fusion, consumers) = plan_fusion(&graph, &shapes, &live, &is_root);

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
                buffer_inputs: Vec::new(),
                out_shape,
                all_same,
                small_members,
                main_members,
            });
        }

        let mut ewise: Vec<Option<(MemberPlan, Vec<usize>)>> = (0..n).map(|_| None).collect();
        for (idx, node) in graph.nodes.iter().enumerate() {
            if !live[idx] || fusion.group_of[idx].is_some() || !is_elementwise(node) {
                continue;
            }
            let target = shapes[idx].clone();
            let inputs: Vec<ChildRef> = node_inputs(node)
                .iter()
                .map(|&c| buffer_child(c, &target))
                .collect();
            ewise[idx] = Some((
                MemberPlan {
                    op: Op::of(node)?,
                    all_same: inputs.iter().all(|c| c.same_shape),
                    inputs,
                    out_shape: target,
                },
                Vec::new(),
            ));
        }

        let target = Target::parse(graph.device.as_deref())?;

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
    let mut plan = PreparedGraph::prepare(graph).map_err(to_napi_err)?;
    localize(&mut plan);
    let prep = Arc::new(plan);
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
/// A prepared plan plus its pinned leaf buffer. Pins are copies owned by
/// the handle (copy-on-pin: a borrowed JS buffer could be collected or
/// detached while rayon reads it), made once at pin time — per eval only
/// the *dirty* leaves are copied again, which is what removes the
/// pack-every-leaf cost from a compiled step whose big captures (edge
/// lists, targets) never change.
struct HandleState {
    prep: Arc<PreparedGraph>,
    leaves: Vec<u8>,
    /// (byte offset, byte length) per JSON `leaf` index.
    offsets: Vec<(usize, usize)>,
}

static PLAN_HANDLES: OnceLock<Mutex<HashMap<u32, HandleState>>> = OnceLock::new();
static NEXT_HANDLE: Mutex<u32> = Mutex::new(1);

fn handles() -> &'static Mutex<HashMap<u32, HandleState>> {
    PLAN_HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn leaf_offsets(prep: &PreparedGraph) -> candle_core::Result<(Vec<(usize, usize)>, usize)> {
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    let mut total = 0usize;
    for node in &prep.graph.nodes {
        if let Node::Leaf {
            leaf,
            offset,
            shape,
            dtype,
        } = node
        {
            let numel = prod(shape);
            let bytes = numel * LeafTy::parse(dtype.as_deref())?.size();
            if offsets.len() <= *leaf {
                offsets.resize(*leaf + 1, (0, 0));
            }
            offsets[*leaf] = (*offset, bytes);
            total = total.max(*offset + bytes);
        }
    }
    Ok((offsets, total))
}

/// Parse and plan a graph once, returning a handle for `evalPrepared`.
#[napi(js_name = "prepareGraph")]
pub fn prepare_graph(graph_json: String) -> Result<u32> {
    let prep = prepared(&graph_json)?;
    let (offsets, total) = leaf_offsets(&prep).map_err(to_napi_err)?;
    let mut next = NEXT_HANDLE.lock().unwrap();
    let handle = *next;
    *next += 1;
    handles().lock().unwrap().insert(
        handle,
        HandleState {
            prep,
            leaves: vec![0u8; total],
            offsets,
        },
    );
    Ok(handle)
}

/// Copy a leaf's current values into the handle's pinned buffer.
#[napi(js_name = "pinLeaf")]
pub fn pin_leaf(handle: u32, leaf: u32, data: Uint8Array) -> Result<()> {
    let mut map = handles().lock().unwrap();
    let state = map.get_mut(&handle).ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("unknown prepared graph {handle}"),
        )
    })?;
    let (offset, bytes) = *state
        .offsets
        .get(leaf as usize)
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("unknown leaf {leaf}")))?;
    if data.len() != bytes {
        return Err(Error::new(
            Status::InvalidArg,
            format!("leaf {leaf} expects {bytes} bytes, got {}", data.len()),
        ));
    }
    state.leaves[offset..offset + bytes].copy_from_slice(&data);
    Ok(())
}

/// Drop a plan created by `prepareGraph`.
#[napi(js_name = "releaseGraph")]
pub fn release_graph(handle: u32) {
    handles().lock().unwrap().remove(&handle);
}

/// How many prepared-graph handles are currently held. Used by tests to
/// assert that `releaseGraph` / compiled `dispose()` actually frees them.
#[napi(js_name = "preparedGraphCount")]
pub fn prepared_graph_count() -> u32 {
    handles().lock().unwrap().len() as u32
}

/// Evaluate a plan: overlay the dirty leaves (packed in increasing leaf
/// index, listed by `dirty_index`) onto the pins, then run. JS callers
/// are single-threaded, so holding the handle lock through the
/// evaluation cannot deadlock; rayon workers never touch the handle map.
#[napi(js_name = "evalPrepared")]
pub fn eval_prepared(
    handle: u32,
    dirty: Uint8Array,
    dirty_index: Uint32Array,
    seed: u32,
) -> Result<Readback> {
    let mut map = handles().lock().unwrap();
    let state = map.get_mut(&handle).ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("unknown prepared graph {handle}"),
        )
    })?;
    let mut cursor = 0usize;
    for &leaf in dirty_index.iter() {
        let (offset, bytes) = *state.offsets.get(leaf as usize).ok_or_else(|| {
            Error::new(Status::InvalidArg, format!("unknown dirty leaf {leaf}"))
        })?;
        let chunk = dirty.get(cursor..cursor + bytes).ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                format!("dirty buffer too short for leaf {leaf}"),
            )
        })?;
        state.leaves[offset..offset + bytes].copy_from_slice(chunk);
        cursor += bytes;
    }
    evaluate(&state.prep, &state.leaves, seed)
}

/// Rewrite every ChildSource::Buffer from a global node index to an
/// index into the plan's own `buffer_inputs` list, so execution packs
/// exactly the inputs a pass reads.
fn localize_members(
    members: &mut [MemberPlan],
    locals: &mut Vec<usize>,
) {
    for m in members {
        for cr in &mut m.inputs {
            if let ChildSource::Buffer(global) = cr.source {
                let local = locals
                    .iter()
                    .position(|&x| x == global)
                    .unwrap_or_else(|| {
                        locals.push(global);
                        locals.len() - 1
                    });
                cr.source = ChildSource::Buffer(local);
            }
        }
    }
}

fn localize(prep: &mut PreparedGraph) {
    for g in &mut prep.groups {
        let mut locals = Vec::new();
        localize_members(&mut g.small_members, &mut locals);
        localize_members(&mut g.main_members, &mut locals);
        g.buffer_inputs = locals;
    }
    for entry in prep.ewise.iter_mut().flatten() {
        let mut locals = Vec::new();
        localize_members(std::slice::from_mut(&mut entry.0), &mut locals);
        entry.1 = locals;
    }
}

/// A loop-evaluator buffer: shared storage plus view metadata. The
/// structural ops (view / permute / narrow / broadcastTo) only rewrite
/// the metadata; a consumer that needs packed row-major data calls
/// `packed()`, which borrows when the view is already contiguous.
#[derive(Clone)]
struct Buf {
    data: Arc<Vec<f32>>,
    offset: usize,
    shape: Vec<usize>,
    /// Element strides; 0 on broadcast dims.
    strides: Vec<usize>,
}

impl Buf {
    fn owned(data: Vec<f32>, shape: Vec<usize>) -> Buf {
        let strides = row_major_strides(&shape);
        Buf {
            data: Arc::new(data),
            offset: 0,
            shape,
            strides,
        }
    }

    fn numel(&self) -> usize {
        prod(&self.shape)
    }

    fn is_contiguous(&self) -> bool {
        self.strides == row_major_strides(&self.shape)
    }

    fn packed(&self) -> std::borrow::Cow<'_, [f32]> {
        let n = self.numel();
        if self.is_contiguous() {
            return std::borrow::Cow::Borrowed(&self.data[self.offset..self.offset + n]);
        }
        // Odometer walk: incremental index updates, and the innermost
        // dim copied as a slice when it is unit-stride.
        let rank = self.shape.len();
        let mut out = vec![0f32; n];
        if rank == 0 {
            out[0] = self.data[self.offset];
            return std::borrow::Cow::Owned(out);
        }
        let inner = self.shape[rank - 1];
        let inner_stride = self.strides[rank - 1];
        let outer = n / inner.max(1);
        let mut coords = vec![0usize; rank.saturating_sub(1)];
        let mut base = self.offset;
        let mut o = 0usize;
        for _ in 0..outer {
            if inner_stride == 1 {
                out[o..o + inner]
                    .copy_from_slice(&self.data[base..base + inner]);
            } else {
                for k in 0..inner {
                    out[o + k] = self.data[base + k * inner_stride];
                }
            }
            o += inner;
            for d in (0..rank - 1).rev() {
                coords[d] += 1;
                base += self.strides[d];
                if coords[d] < self.shape[d] {
                    break;
                }
                base -= self.strides[d] * self.shape[d];
                coords[d] = 0;
            }
        }
        std::borrow::Cow::Owned(out)
    }
}

#[inline]
fn read_ref(
    cr: &ChildRef,
    i: usize,
    coords: &[usize],
    inputs: &[&[f32]],
    temps: &[Vec<f32>],
    scratch: &[f32],
) -> f32 {
    match cr.source {
        ChildSource::Slot(slot) => scratch[slot],
        ChildSource::Temp(t) => read_bcast(&temps[t], &cr.strides, cr.same_shape, i, coords),
        ChildSource::Buffer(b) => read_bcast(
            inputs[b],
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
    inputs: &[&[f32]],
    temps: &[Vec<f32>],
    scratch: &[f32],
) -> f32 {
    match cr.source {
        ChildSource::Slot(slot) => scratch[slot],
        ChildSource::Temp(t) => temps[t][i],
        ChildSource::Buffer(b) => inputs[b][i],
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
    inputs_slices: &[&[f32]],
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
                        read_flat(&inputs[0], i, inputs_slices, temps, &[]),
                        read_flat(&inputs[1], i, inputs_slices, temps, &[]),
                    ),
                    Op::Un(kind, p) => apply_un(
                        kind,
                        p,
                        read_flat(&inputs[0], i, inputs_slices, temps, &[]),
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
                    read_ref(&inputs[0], i, &coords, inputs_slices, temps, &[]),
                    read_ref(&inputs[1], i, &coords, inputs_slices, temps, &[]),
                ),
                Op::Un(kind, p) => apply_un(
                    kind,
                    p,
                    read_ref(&inputs[0], i, &coords, inputs_slices, temps, &[]),
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
fn exec_group(plan: &GroupPlan, inputs_slices: &[&[f32]]) -> Vec<f32> {
    // Members smaller than the output shape are evaluated first, into
    // temps: their inputs can only be external buffers or earlier temps.
    let mut temps: Vec<Vec<f32>> = Vec::with_capacity(plan.small_members.len());
    for sm in &plan.small_members {
        temps.push(exec_member(sm, inputs_slices, &temps));
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
                            read_flat(&mm.inputs[0], i, inputs_slices, &temps, &scratch),
                            read_flat(&mm.inputs[1], i, inputs_slices, &temps, &scratch),
                        ),
                        Op::Un(kind, p) => apply_un(
                            kind,
                            p,
                            read_flat(&mm.inputs[0], i, inputs_slices, &temps, &scratch),
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
                        read_ref(&mm.inputs[0], i, &coords, inputs_slices, &temps, &scratch),
                        read_ref(&mm.inputs[1], i, &coords, inputs_slices, &temps, &scratch),
                    ),
                    Op::Un(kind, p) => apply_un(
                        kind,
                        p,
                        read_ref(&mm.inputs[0], i, &coords, inputs_slices, &temps, &scratch),
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
fn execute(prep: &PreparedGraph, leaves: &[u8], seed: u32) -> candle_core::Result<Vec<f32>> {
    let graph = &prep.graph;
    let n = graph.nodes.len();
    let mut buffers: Vec<Option<Buf>> = (0..n).map(|_| None).collect();
    // The same consumer countdown run_graph does: drop a buffer once
    // nothing else will read it, so a long rollout does not hold every
    // activation it ever produced. Views share the Arc, so storage is
    // freed when the last view dies.
    let mut remaining = prep.consumers.clone();
    let mut members_of: Vec<Vec<usize>> = vec![Vec::new(); prep.groups.len()];
    for i in 0..n {
        if let Some(g) = prep.group_of[i] {
            members_of[g].push(i);
        }
    }
    let release = |remaining: &mut Vec<usize>,
                   buffers: &mut Vec<Option<Buf>>,
                   is_root: &[bool],
                   input: usize| {
        remaining[input] -= 1;
        if remaining[input] == 0 && !is_root[input] {
            buffers[input] = None;
        }
    };
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !prep.live[idx] {
            continue;
        }
        if let Some(g) = prep.group_of[idx] {
            if prep.groups[g].leader == idx {
                let plan = &prep.groups[g];
                let packed: Vec<std::borrow::Cow<[f32]>> = plan
                    .buffer_inputs
                    .iter()
                    .map(|&gi| buffers[gi].as_ref().expect("group input computed").packed())
                    .collect();
                let slices: Vec<&[f32]> = packed.iter().map(|c| c.as_ref()).collect();
                let out = exec_group(plan, &slices);
                drop(packed);
                buffers[idx] = Some(Buf::owned(out, prep.shapes[idx].clone()));
                for &m in &members_of[g] {
                    for input in node_inputs(&graph.nodes[m]) {
                        release(&mut remaining, &mut buffers, &prep.is_root, input);
                    }
                }
            }
            continue;
        }
        let started = if profiling() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let get = |i: usize| -> candle_core::Result<&Buf> {
            buffers.get(i).and_then(|b| b.as_ref()).ok_or_else(|| {
                candle_core::Error::Msg(format!("node references future index {i}"))
            })
        };
        let out: Buf = match node {
            Node::Leaf {
                leaf,
                offset,
                shape,
                dtype,
            } => {
                let n = prod(shape);
                let ty = LeafTy::parse(dtype.as_deref())?;
                Buf::owned(
                    read_leaf_f32(leaves, *leaf, *offset, n, ty)?,
                    shape.clone(),
                )
            }
            Node::Binary { .. } | Node::Unary { .. } => {
                let (plan, locals) = prep.ewise[idx].as_ref().unwrap();
                let packed: Vec<std::borrow::Cow<[f32]>> = locals
                    .iter()
                    .map(|&gi| buffers[gi].as_ref().expect("ewise input computed").packed())
                    .collect();
                let slices: Vec<&[f32]> = packed.iter().map(|c| c.as_ref()).collect();
                Buf::owned(
                    exec_member(plan, &slices, &[]),
                    prep.shapes[idx].clone(),
                )
            }
            Node::Matmul { a, b, .. } => Buf::owned(
                cpu_matmul(
                    &get(*a)?.packed(),
                    &prep.shapes[*a],
                    &get(*b)?.packed(),
                    &prep.shapes[*b],
                )?,
                prep.shapes[idx].clone(),
            ),
            Node::Reduce {
                kind,
                dim,
                keepdim,
                input,
                ..
            } => Buf::owned(
                tiny_reduce(kind, *dim, *keepdim, &get(*input)?.packed(), &prep.shapes[*input])?,
                prep.shapes[idx].clone(),
            ),
            Node::ReduceAll { kind, input, .. } => Buf::owned(
                tiny_reduce_all(kind, &get(*input)?.packed())?,
                prep.shapes[idx].clone(),
            ),
            // The structural ops are metadata rewrites: no copy here. A
            // consumer that needs packed data pays for it there, and a
            // chain of views/narrows/permutes collapses to arithmetic.
            Node::BroadcastTo { input, shape } => {
                let src = get(*input)?;
                let rank = shape.len();
                let offset_dims = rank - src.shape.len();
                let mut strides = vec![0usize; rank];
                for j in 0..src.shape.len() {
                    strides[offset_dims + j] = if src.shape[j] == 1 && shape[offset_dims + j] != 1 {
                        0
                    } else {
                        src.strides[j]
                    };
                }
                Buf {
                    data: src.data.clone(),
                    offset: src.offset,
                    shape: shape.clone(),
                    strides,
                }
            }
            Node::Permute { order, input, .. } => {
                let src = get(*input)?;
                Buf {
                    data: src.data.clone(),
                    offset: src.offset,
                    shape: order.iter().map(|&d| src.shape[d]).collect(),
                    strides: order.iter().map(|&d| src.strides[d]).collect(),
                }
            }
            Node::View { input, shape } => {
                let src = get(*input)?;
                if src.is_contiguous() {
                    Buf {
                        data: src.data.clone(),
                        offset: src.offset,
                        shape: shape.clone(),
                        strides: row_major_strides(shape),
                    }
                } else {
                    Buf::owned(src.packed().into_owned(), shape.clone())
                }
            }
            Node::Narrow {
                dim,
                start,
                length,
                input,
                ..
            } => {
                let src = get(*input)?;
                let mut shape = src.shape.clone();
                shape[*dim] = *length;
                Buf {
                    data: src.data.clone(),
                    offset: src.offset + start * src.strides[*dim],
                    shape,
                    strides: src.strides.clone(),
                }
            }
            Node::Cat { a, b, dim, .. } => Buf::owned(
                tiny_cat(
                    &get(*a)?.packed(),
                    &prep.shapes[*a],
                    &get(*b)?.packed(),
                    &prep.shapes[*b],
                    *dim,
                ),
                prep.shapes[idx].clone(),
            ),
            Node::OneHot { classes, input, .. } => Buf::owned(
                tiny_one_hot(*classes, &get(*input)?.packed())?,
                prep.shapes[idx].clone(),
            ),
            Node::IndexSelect { dim, input, index, .. } => Buf::owned(
                tiny_index_select(
                    &get(*input)?.packed(),
                    &prep.shapes[*input],
                    &get(*index)?.packed(),
                    *dim,
                )?,
                prep.shapes[idx].clone(),
            ),
            Node::ScatterAdd {
                dim,
                length,
                input,
                index,
                ..
            } => Buf::owned(
                tiny_scatter_add(
                    &get(*input)?.packed(),
                    &prep.shapes[*input],
                    &get(*index)?.packed(),
                    *dim,
                    *length,
                )?,
                prep.shapes[idx].clone(),
            ),
            Node::Random {
                kind,
                stream,
                shape,
            } => Buf::owned(random_data(kind, prod(shape), *stream, seed)?, shape.clone()),
        };
        if let Some(started) = started {
            record(
                op_kind(node),
                started.elapsed().as_secs_f64(),
                prod(&prep.shapes[idx]),
            );
        }
        buffers[idx] = Some(out);
        for input in node_inputs(node) {
            release(&mut remaining, &mut buffers, &prep.is_root, input);
        }
    }
    let mut out = Vec::new();
    for &r in &prep.roots {
        let buf = buffers.get(r).and_then(|b| b.as_ref()).ok_or_else(|| {
            candle_core::Error::Msg(format!("root references missing node {r}"))
        })?;
        out.extend_from_slice(&buf.packed());
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

    // A single slice — the usual dim-0 aggregation over an edge list.
    // One serial pass that walks the edges exactly once. The old
    // per-thread block scan re-read the whole index once per thread
    // (threads x E index reads); profiled on the GNCA mix it lost to
    // even candle's serial index_add, so the simple loop wins until a
    // sort-by-destination segmented reduce is worth its setup.
    for (j, &row) in indices.iter().enumerate() {
        let to = row * inner;
        let from = j * inner;
        for k in 0..inner {
            out[to + k] += data[from + k];
        }
    }
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

fn validate_one_hot_targets(classes: usize, values: &[f32]) -> candle_core::Result<()> {
    for &v in values {
        if v.fract() != 0.0 || v < 0.0 || v >= classes as f32 {
            return Err(candle_core::Error::Msg(format!(
                "oneHot: target {v} out of range for {classes} classes"
            )));
        }
    }
    Ok(())
}

fn tiny_one_hot(classes: usize, data: &[f32]) -> candle_core::Result<Vec<f32>> {
    validate_one_hot_targets(classes, data)?;
    let mut out = vec![0f32; data.len() * classes];
    for (i, &v) in data.iter().enumerate() {
        out[i * classes + v as usize] = 1.0;
    }
    Ok(out)
}

// Candle's dim-reductions (sum/max/min) squeeze the dim; typenet's
// keepdim semantics require re-inserting it.
fn reinsert_dim(t: Tensor, dim: usize, keepdim: bool) -> candle_core::Result<Tensor> {
    if keepdim { t.unsqueeze(dim) } else { Ok(t) }
}

/// Summing away the outer dim of a row-major matrix is a sequential read
/// with one accumulator per column, but candle does it at 801 M elem/s
/// against 2521 for the inner dim — the wrong way round. A row of ones
/// times the matrix is the same sum through Accelerate at 3295, so above a
/// size where the setup is noise, take that route. It reassociates the
/// summation, which for f32 is a difference of about 1e-6 relative, and
/// blocked accumulation is if anything the more accurate of the two.
///
/// This is the bias-gradient shape, so it shows up twice per rolled-out
/// step of a message-passing rollout.
const GEMV_SUM_MIN_ROWS: usize = 4096;

fn eval_reduce(
    kind: &str,
    dim: usize,
    keepdim: bool,
    a: &Tensor,
    ones: &mut HashMap<usize, Tensor>,
) -> candle_core::Result<Tensor> {
    match kind {
        "sum"
            if dim == 0
                && a.rank() == 2
                && a.dim(0)? >= GEMV_SUM_MIN_ROWS =>
        {
            let rows = a.dim(0)?;
            let row = match ones.get(&rows) {
                Some(row) => row.clone(),
                None => {
                    let row = Tensor::ones((1, rows), DType::F32, a.device())?;
                    ones.insert(rows, row.clone());
                    row
                }
            };
            let summed = row.matmul(&a.contiguous()?)?;
            if keepdim {
                Ok(summed)
            } else {
                summed.reshape(a.dim(1)?)
            }
        }
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
    validate_one_hot_targets(classes, &values)?;
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

/// Index tensors arrive as f32 (integral values, exact to 16.7M rows)
/// or as int32/int64 (exact across the full integer range). candle's
/// gather/scatter kernels want U32, so cast; the integer paths skip the
/// f32 mantissa limit entirely.
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
fn scalar_leaf(graph: &Graph, leaves: &[u8], at: usize) -> Option<f32> {
    match &graph.nodes[at] {
        Node::Leaf {
            leaf,
            offset,
            shape,
            dtype,
        } if prod(shape) == 1 => {
            // Only f32 one-element leaves are constant-folded; an int
            // scalar never lands here (indices are never constants).
            if LeafTy::parse(dtype.as_deref()).ok()? != LeafTy::F32 {
                return None;
            }
            let bytes = leaf_bytes(leaves, *leaf, *offset, 1, 4).ok()?;
            Some(f32::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]))
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
    leaves: &[u8],
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
    // gather or scatter — and each read would otherwise re-run the dtype
    // cast to u32. Convert once and keep it for as long as the original
    // is alive.
    let mut indices: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    // Rows of ones for the gemv-style sums below, one per width needed.
    let mut ones: HashMap<usize, Tensor> = HashMap::new();
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
                dtype,
            } => {
                let n = prod(shape);
                let ty = LeafTy::parse(dtype.as_deref())?;
                match ty {
                    LeafTy::F32 => Tensor::from_vec(
                        read_leaf_f32(leaves, *leaf, *offset, n, ty)?,
                        shape.clone(),
                        device,
                    )?,
                    LeafTy::I32 => Tensor::from_vec(
                        read_leaf_i32(leaves, *leaf, *offset, n)?,
                        shape.clone(),
                        device,
                    )?,
                    LeafTy::I64 => Tensor::from_vec(
                        read_leaf_i64(leaves, *leaf, *offset, n)?,
                        shape.clone(),
                        device,
                    )?,
                }
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
                ..
            } => eval_unary(kind, *parameter, get(*input)?)?,
            Node::Matmul { a, b, .. } => {
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
                ..
            } => eval_reduce(kind, *dim, *keepdim, get(*input)?, &mut ones)?,
            Node::ReduceAll { kind, input, .. } => {
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
            Node::Permute { order, input, .. } => get(*input)?.permute(order.clone())?,
            Node::View { input, shape } => {
                get(*input)?.contiguous()?.reshape(shape.clone())?
            }
            Node::Narrow {
                dim,
                start,
                length,
                input,
                ..
            } => get(*input)?.narrow(*dim, *start, *length)?.contiguous()?,
            Node::Cat { a, b, dim, .. } => {
                let a = get(*a)?.contiguous()?;
                let b = get(*b)?.contiguous()?;
                Tensor::cat(&[&a, &b], *dim)?
            }
            Node::OneHot { classes, input, .. } => eval_one_hot(*classes, get(*input)?)?,
            Node::IndexSelect { dim, input, index, .. } => {
                let keys = cached_index(&mut indices, *index, get(*index)?)?;
                get(*input)?.contiguous()?.index_select(&keys, *dim)?
            }
            Node::ScatterAdd {
                dim,
                length,
                input,
                index,
                ..
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
fn forced_target() -> candle_core::Result<Option<Target>> {
    static CHOICE: OnceLock<Option<Target>> = OnceLock::new();
    if let Some(target) = CHOICE.get() {
        return Ok(*target);
    }
    let parsed = match std::env::var("TYPENET_EVALUATOR") {
        Ok(name) if !name.is_empty() => Target::parse(Some(name.as_str())).map(Some)?,
        _ => None,
    };
    Ok(*CHOICE.get_or_init(|| parsed))
}

/// Row-major C = A·B for one packed f32 pair — the eager fast path's
/// escape hatch into Accelerate. Same `gemm` the loop evaluator uses.
#[napi(js_name = "sgemm")]
pub fn sgemm_entry(
    a: Float32Array,
    b: Float32Array,
    m: u32,
    k: u32,
    n: u32,
) -> Result<Readback> {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    if a.len() != m * k || b.len() != k * n {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "sgemm: got {}x{} and {}x{} buffers of {} and {}",
                m, k, k, n,
                a.len(),
                b.len()
            ),
        ));
    }
    let mut c = vec![0f32; m * n];
    gemm(&a, &b, &mut c, m, k, n);
    Ok(vec_readback(c))
}

/// Run a prepared graph on the evaluator it was planned for.
fn evaluate(prep: &PreparedGraph, leaves: &[u8], seed: u32) -> Result<Readback> {
    let target = forced_target().map_err(to_napi_err)?.unwrap_or(prep.target);
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
