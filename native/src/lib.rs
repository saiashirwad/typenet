use candle_core::{DType, Device, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

fn to_napi_err(err: candle_core::Error) -> Error {
    Error::new(Status::GenericFailure, err.to_string())
}

#[derive(Default)]
struct Counters {
    prepares: AtomicU64,
    index_builds: AtomicU64,
    instrs: AtomicU64,
    fused_regions: AtomicU64,
    fusion_matches: AtomicU64,
    gemm_calls: AtomicU64,
    rowwise_calls: AtomicU64,
    candle_dispatches: AtomicU64,
    program_cache_hits: AtomicU64,
    program_cache_misses: AtomicU64,
    program_cache_evictions: AtomicU64,
    prepare_ns: AtomicU64,
    eval_ns: AtomicU64,
}

static COUNTERS: Counters = Counters {
    prepares: AtomicU64::new(0),
    index_builds: AtomicU64::new(0),
    instrs: AtomicU64::new(0),
    fused_regions: AtomicU64::new(0),
    fusion_matches: AtomicU64::new(0),
    gemm_calls: AtomicU64::new(0),
    rowwise_calls: AtomicU64::new(0),
    candle_dispatches: AtomicU64::new(0),
    program_cache_hits: AtomicU64::new(0),
    program_cache_misses: AtomicU64::new(0),
    program_cache_evictions: AtomicU64::new(0),
    prepare_ns: AtomicU64::new(0),
    eval_ns: AtomicU64::new(0),
};

fn program_count() -> u64 {
    PLAN_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .len() as u64
}

// Not yet measurable: always -1, never 0, so callers can tell the two apart.
fn unmeasured() -> serde_json::Value {
    serde_json::Value::from(-1i64)
}

#[napi(js_name = "counters")]
pub fn counters() -> String {
    let mut m: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    m.insert("prepares".into(), COUNTERS.prepares.load(Ordering::Relaxed).into());
    m.insert("indexBuilds".into(), COUNTERS.index_builds.load(Ordering::Relaxed).into());
    m.insert("instrs".into(), COUNTERS.instrs.load(Ordering::Relaxed).into());
    m.insert("fusedRegions".into(), COUNTERS.fused_regions.load(Ordering::Relaxed).into());
    m.insert("gemmCalls".into(), COUNTERS.gemm_calls.load(Ordering::Relaxed).into());
    m.insert("rowwiseCalls".into(), COUNTERS.rowwise_calls.load(Ordering::Relaxed).into());
    m.insert("csrBuilds".into(), unmeasured());
    m.insert("arenaBytes".into(), unmeasured());
    m.insert("peakLiveBytes".into(), unmeasured());
    m.insert("allocationsDuringRun".into(), unmeasured());
    m.insert("residentSlots".into(), unmeasured());
    m.insert("candleDispatches".into(), COUNTERS.candle_dispatches.load(Ordering::Relaxed).into());
    m.insert("programCacheHits".into(), COUNTERS.program_cache_hits.load(Ordering::Relaxed).into());
    m.insert("programCacheMisses".into(), COUNTERS.program_cache_misses.load(Ordering::Relaxed).into());
    m.insert(
        "programCacheEvictions".into(),
        COUNTERS.program_cache_evictions.load(Ordering::Relaxed).into(),
    );
    m.insert("programs".into(), program_count().into());
    let mut match_counts: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    match_counts.insert("fusion".into(), COUNTERS.fusion_matches.load(Ordering::Relaxed).into());
    m.insert("matchCounts".into(), serde_json::to_value(&match_counts).unwrap());
    let mut phase_ns: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    phase_ns.insert("prepare".into(), COUNTERS.prepare_ns.load(Ordering::Relaxed).into());
    phase_ns.insert("eval".into(), COUNTERS.eval_ns.load(Ordering::Relaxed).into());
    m.insert("phaseNs".into(), serde_json::to_value(&phase_ns).unwrap());
    m.insert("storeSlots".into(), unmeasured());
    serde_json::to_string(&m).unwrap()
}

struct Switches {
    no_fusion: bool,
    parallel_min: Option<usize>,
    chunk: Option<usize>,
}

impl Switches {
    fn load() -> Self {
        let flag = |name: &str| std::env::var(name).map(|v| v == "1").unwrap_or(false);
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok());
        Switches {
            no_fusion: flag("TYPENET_NO_FUSION"),
            parallel_min: num("TYPENET_PARALLEL_MIN"),
            chunk: num("TYPENET_CHUNK"),
        }
    }
}

fn switches() -> &'static Switches {
    static SWITCHES: OnceLock<Switches> = OnceLock::new();
    SWITCHES.get_or_init(Switches::load)
}

#[derive(Serialize)]
struct SwitchInfo {
    value: serde_json::Value,
    wired: bool,
}

fn opt_num(v: Option<usize>) -> serde_json::Value {
    v.map(|v| serde_json::Value::from(v as u64)).unwrap_or(serde_json::Value::Null)
}

#[napi(js_name = "deviceInfo")]
pub fn device_info() -> String {
    let s = switches();
    let mut out: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    out.insert("device".into(), serde_json::Value::String(device_name()));
    let mut sw: BTreeMap<String, SwitchInfo> = BTreeMap::new();
    sw.insert("TYPENET_NO_FUSION".into(), SwitchInfo { value: s.no_fusion.into(), wired: true });
    sw.insert("TYPENET_PARALLEL_MIN".into(), SwitchInfo { value: opt_num(s.parallel_min), wired: true });
    sw.insert("TYPENET_CHUNK".into(), SwitchInfo { value: opt_num(s.chunk), wired: true });
    out.insert("switches".into(), serde_json::to_value(&sw).unwrap());
    serde_json::to_string(&out).unwrap()
}

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

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum Node {
    Leaf {
        leaf: usize,
        offset: usize,
        shape: Vec<usize>,
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
    IndexSelect {
        dim: usize,
        input: usize,
        index: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    ScatterAdd {
        dim: usize,
        length: usize,
        input: usize,
        index: usize,
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    Random {
        kind: String,
        stream: u32,
        shape: Vec<usize>,
    },
}

#[derive(Debug, Deserialize)]
struct Graph {
    nodes: Vec<Node>,
    #[serde(default)]
    roots: Option<Vec<usize>>,
    #[serde(default)]
    device: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    Loops,
    Cpu,
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

fn leaf_bytes(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
    width: usize,
) -> candle_core::Result<&[u8]> {
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

trait LeafScalar: Copy {
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl LeafScalar for f32 {
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes.try_into().expect("f32 is 4 bytes"))
    }
}

impl LeafScalar for i32 {
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_ne_bytes(bytes.try_into().expect("i32 is 4 bytes"))
    }
}

impl LeafScalar for i64 {
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_ne_bytes(bytes.try_into().expect("i64 is 8 bytes"))
    }
}

fn decode_le<T: LeafScalar>(bytes: &[u8], n: usize) -> Vec<T> {
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .take(n)
        .map(T::from_le_bytes)
        .collect()
}

fn read_leaf_f32(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
    ty: LeafTy,
) -> candle_core::Result<Vec<f32>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, ty.size())?;
    Ok(match ty {
        LeafTy::F32 => decode_le::<f32>(bytes, n),
        LeafTy::I32 => decode_le::<i32>(bytes, n)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        LeafTy::I64 => decode_le::<i64>(bytes, n)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    })
}

fn read_leaf_i32(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
) -> candle_core::Result<Vec<i32>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, 4)?;
    Ok(decode_le::<i32>(bytes, n))
}

fn read_leaf_i64(
    leaves: &[u8],
    leaf: usize,
    offset: usize,
    n: usize,
) -> candle_core::Result<Vec<i64>> {
    let bytes = leaf_bytes(leaves, leaf, offset, n, 8)?;
    Ok(decode_le::<i64>(bytes, n))
}

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

// Counter-based RNG matching src/kernels.ts: uniforms agree bit-for-bit, normals
// to f32 rounding. The seed travels per eval, so a cached plan stays valid.
#[inline]
fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

#[inline]
fn unit_float(seed: u32, stream: u32, i: u32) -> f32 {
    let mixed = hash32(hash32(seed ^ stream.wrapping_mul(0x9e37_79b9)) ^ i);
    (mixed >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn random_data(kind: &str, n: usize, stream: u32, seed: u32) -> candle_core::Result<Vec<f32>> {
    match kind {
        "uniform" => Ok((0..n).map(|i| unit_float(seed, stream, i as u32)).collect()),
        "normal" => Ok((0..n)
            .map(|i| {
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
        "reluGrad" => elementwise(a, b, |x, y| x * &y.sign()?.relu()?),
        "leakyReluGrad" => elementwise(a, b, |x, y| {
            let m = mask_f32(&y.gt(&y.zeros_like()?)?)?;
            let weights = (&m + &(m.ones_like()? - &m)? * parameter)?;
            x * &weights
        }),
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
        "leakyRelu" => a.relu()? - (a.neg()?.relu()? * parameter)?,
        "sigmoid" => a.affine(0.5, 0.0)?.tanh()?.affine(0.5, 0.5),
        "tanh" => a.tanh(),
        "scalePowGrad" => a.powf(parameter - 1.0)? * parameter,
        other => Err(candle_core::Error::Msg(format!(
            "unknown unary op: {other}"
        ))),
    }
}

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

macro_rules! binary_ops {
    ($($name:literal => $variant:path => $apply:expr),* $(,)?) => {
        fn parse(kind: &str) -> candle_core::Result<Self> {
            Ok(match kind {
                $($name => $variant,)*
                other => {
                    return Err(candle_core::Error::Msg(format!(
                        "unknown binary op: {other}"
                    )))
                }
            })
        }

        #[inline(always)]
        fn apply(kind: Self, p: f32, a: f32, b: f32) -> f32 {
            match kind {
                $($variant => ($apply)(a, b, p),)*
            }
        }

        #[cfg(test)]
        fn all() -> &'static [&'static str] {
            &[$($name,)*]
        }
    };
}

impl Bin {
    binary_ops! {
        "add" => Bin::Add => |a: f32, b: f32, _p: f32| a + b,
        "sub" => Bin::Sub => |a: f32, b: f32, _p: f32| a - b,
        "mul" => Bin::Mul => |a: f32, b: f32, _p: f32| a * b,
        "div" => Bin::Div => |a: f32, b: f32, _p: f32| a / b,
        "maximum" => Bin::Maximum => |a: f32, b: f32, _p: f32| { if a >= b { a } else { b } },
        "minimum" => Bin::Minimum => |a: f32, b: f32, _p: f32| { if a <= b { a } else { b } },
        "gt" => Bin::Gt => |a: f32, b: f32, _p: f32| (a > b) as u8 as f32,
        "ge" => Bin::Ge => |a: f32, b: f32, _p: f32| (a >= b) as u8 as f32,
        "lt" => Bin::Lt => |a: f32, b: f32, _p: f32| (a < b) as u8 as f32,
        "le" => Bin::Le => |a: f32, b: f32, _p: f32| (a <= b) as u8 as f32,
        "eq" => Bin::Eq => |a: f32, b: f32, _p: f32| (a == b) as u8 as f32,
        "negDiv" => Bin::NegDiv => |a: f32, b: f32, _p: f32| -a / b,
        "halfDiv" => Bin::HalfDiv => |a: f32, b: f32, _p: f32| 0.5 * a / b,
        "mulSign" => Bin::MulSign => |a: f32, b: f32, _p: f32| a * ((b > 0.0) as u8 as f32 - (b < 0.0) as u8 as f32),
        "reluGrad" => Bin::ReluGrad => |a: f32, b: f32, _p: f32| { if b > 0.0 { a } else { 0.0 } },
        "leakyReluGrad" => Bin::LeakyReluGrad => |a: f32, b: f32, p: f32| a * if b > 0.0 { 1.0 } else { p },
        "sigmoidGrad" => Bin::SigmoidGrad => |a: f32, b: f32, _p: f32| a * b * (1.0 - b),
        "tanhGrad" => Bin::TanhGrad => |a: f32, b: f32, _p: f32| a * (1.0 - b * b),
    }
}

macro_rules! unary_ops {
    ($($name:literal => $variant:path => $apply:expr),* $(,)?) => {
        fn parse(kind: &str) -> candle_core::Result<Self> {
            Ok(match kind {
                $($name => $variant,)*
                other => {
                    return Err(candle_core::Error::Msg(format!(
                        "unknown unary op: {other}"
                    )))
                }
            })
        }

        #[inline(always)]
        fn apply(kind: Self, p: f32, x: f32) -> f32 {
            match kind {
                $($variant => ($apply)(x, p),)*
            }
        }

        #[cfg(test)]
        fn all() -> &'static [&'static str] {
            &[$($name,)*]
        }
    };
}

impl Un {
    unary_ops! {
        "pow" => Un::Pow => |x: f32, p: f32| x.powf(p),
        "neg" => Un::Neg => |x: f32, _p: f32| -x,
        "exp" => Un::Exp => |x: f32, _p: f32| x.exp(),
        "log" => Un::Log => |x: f32, _p: f32| x.ln(),
        "sqrt" => Un::Sqrt => |x: f32, _p: f32| x.sqrt(),
        "abs" => Un::Abs => |x: f32, _p: f32| x.abs(),
        "relu" => Un::Relu => |x: f32, _p: f32| x.max(0.0),
        "leakyRelu" => Un::LeakyRelu => |x: f32, p: f32| { if x > 0.0 { x } else { p * x } },
        "sigmoid" => Un::Sigmoid => |x: f32, _p: f32| 1.0 / (1.0 + (-x).exp()),
        "tanh" => Un::Tanh => |x: f32, _p: f32| x.tanh(),
        "scalePowGrad" => Un::ScalePowGrad => |x: f32, p: f32| p * x.powf(p - 1.0),
    }
}

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
    Bin::apply(kind, p, a, b)
}

#[inline(always)]
fn apply_un(kind: Un, p: f32, x: f32) -> f32 {
    Un::apply(kind, p, x)
}

// The loops evaluator exists because candle's per-op dispatch dominates on tiny graphs.
// A node joins its consumer's fusion group only if it is elementwise, live,
// broadcastable to the group's output shape, single-consumer, and not a root.
struct FusionPlan {
    group_of: Vec<Option<usize>>,
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
    let mut consumers = vec![0usize; n];
    for (i, node) in graph.nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        for input in node_inputs(node) {
            consumers[input] += 1;
        }
    }
    if switches().no_fusion {
        return (FusionPlan { group_of: vec![None; n], groups: Vec::new() }, consumers);
    }
    let mut group_of: Vec<Option<usize>> = vec![None; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
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
            group_of[leader] = None;
        } else {
            members.sort_unstable();
            groups.push(members);
        }
    }
    (FusionPlan { group_of, groups }, consumers)
}

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

enum ChildSource {
    Buffer(usize),
    Temp(usize),
    Slot(usize),
}

struct ChildRef {
    source: ChildSource,
    strides: Vec<usize>,
    same_shape: bool,
}

struct MemberPlan {
    op: Op,
    out_shape: Vec<usize>,
    inputs: Vec<ChildRef>,
    all_same: bool,
}

struct GroupPlan {
    leader: usize,
    buffer_inputs: Vec<usize>,
    out_shape: Vec<usize>,
    all_same: bool,
    small_members: Vec<MemberPlan>,
    main_members: Vec<MemberPlan>,
}

struct PreparedGraph {
    graph: Graph,
    shapes: Vec<Vec<usize>>,
    roots: Vec<usize>,
    live: Vec<bool>,
    consumers: Vec<usize>,
    is_root: Vec<bool>,
    group_of: Vec<Option<usize>>,
    groups: Vec<GroupPlan>,
    ewise: Vec<Option<(MemberPlan, Vec<usize>)>>,
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

fn prepared(graph_json: &str) -> Result<Arc<PreparedGraph>> {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = cache.lock().unwrap().get(graph_json) {
        COUNTERS.program_cache_hits.fetch_add(1, Ordering::Relaxed);
        return Ok(p.clone());
    }
    COUNTERS.program_cache_misses.fetch_add(1, Ordering::Relaxed);
    let graph: Graph = serde_json::from_str(graph_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("invalid graph JSON: {e}")))?;
    let mut plan = PreparedGraph::prepare(graph).map_err(to_napi_err)?;
    localize(&mut plan);
    let live_instrs = plan.live.iter().filter(|&&l| l).count() as u64;
    COUNTERS.instrs.fetch_add(live_instrs, Ordering::Relaxed);
    COUNTERS.fused_regions.fetch_add(plan.groups.len() as u64, Ordering::Relaxed);
    let matched: u64 = plan
        .groups
        .iter()
        .map(|g| (g.small_members.len() + g.main_members.len()).saturating_sub(1) as u64)
        .sum();
    COUNTERS.fusion_matches.fetch_add(matched, Ordering::Relaxed);
    let prep = Arc::new(plan);
    let mut map = cache.lock().unwrap();
    if map.len() >= 128 {
        COUNTERS.program_cache_evictions.fetch_add(map.len() as u64, Ordering::Relaxed);
        map.clear();
    }
    map.insert(graph_json.to_string(), prep.clone());
    Ok(prep)
}

// Pins are copies owned by the handle: a borrowed JS buffer could be collected
// or detached while rayon reads it, and only dirty leaves are copied per eval.
struct HandleState {
    prep: Arc<PreparedGraph>,
    leaves: Vec<u8>,
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

#[napi(js_name = "prepareGraph")]
pub fn prepare_graph(graph_json: String) -> Result<u32> {
    let started = std::time::Instant::now();
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
    COUNTERS.prepares.fetch_add(1, Ordering::Relaxed);
    COUNTERS.prepare_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(handle)
}

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

#[napi(js_name = "releaseGraph")]
pub fn release_graph(handle: u32) {
    handles().lock().unwrap().remove(&handle);
}

#[napi(js_name = "preparedGraphCount")]
pub fn prepared_graph_count() -> u32 {
    handles().lock().unwrap().len() as u32
}

// JS callers are single-threaded, so holding the handle lock through evaluation cannot deadlock.
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

#[derive(Clone)]
struct Buf {
    data: Arc<Vec<f32>>,
    offset: usize,
    shape: Vec<usize>,
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

const CHUNK_DEFAULT: usize = 8192;

const PARALLEL_MIN_DEFAULT: usize = 16384;

fn chunk_size() -> usize {
    switches().chunk.unwrap_or(CHUNK_DEFAULT)
}

fn parallel_min() -> usize {
    switches().parallel_min.unwrap_or(PARALLEL_MIN_DEFAULT)
}

fn over_chunks(out: &mut [f32], body: impl Fn(usize, &mut [f32]) + Send + Sync) {
    if out.len() < parallel_min() {
        body(0, out);
        return;
    }
    let chunk = chunk_size();
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c, slice)| body(c * chunk, slice));
}

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

fn exec_group(plan: &GroupPlan, inputs_slices: &[&[f32]]) -> Vec<f32> {
    let mut temps: Vec<Vec<f32>> = Vec::with_capacity(plan.small_members.len());
    for sm in &plan.small_members {
        temps.push(exec_member(sm, inputs_slices, &temps));
    }
    let members = &plan.main_members;
    let last = members.len() - 1;
    let shape = &plan.out_shape;
    let mut out = vec![0f32; prod(shape)];
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

#[inline]
fn release_input(
    remaining: &mut [usize],
    is_root: &[bool],
    input: usize,
    mut drop_buf: impl FnMut(usize),
) {
    remaining[input] -= 1;
    if remaining[input] == 0 && !is_root[input] {
        drop_buf(input);
    }
}

fn execute(prep: &PreparedGraph, leaves: &[u8], seed: u32) -> candle_core::Result<Vec<f32>> {
    let graph = &prep.graph;
    let n = graph.nodes.len();
    let mut buffers: Vec<Option<Buf>> = (0..n).map(|_| None).collect();
    let mut remaining = prep.consumers.clone();
    let mut members_of: Vec<Vec<usize>> = vec![Vec::new(); prep.groups.len()];
    for i in 0..n {
        if let Some(g) = prep.group_of[i] {
            members_of[g].push(i);
        }
    }
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
                        release_input(&mut remaining, &prep.is_root, input, |i| {
                            buffers[i] = None;
                        });
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
            Node::Matmul { a, b, .. } => {
                COUNTERS.gemm_calls.fetch_add(1, Ordering::Relaxed);
                Buf::owned(
                    cpu_matmul(
                        &get(*a)?.packed(),
                        &prep.shapes[*a],
                        &get(*b)?.packed(),
                        &prep.shapes[*b],
                    )?,
                    prep.shapes[idx].clone(),
                )
            }
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
            release_input(&mut remaining, &prep.is_root, input, |i| {
                buffers[i] = None;
            });
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

#[cfg(target_os = "macos")]
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if c.is_empty() || k == 0 {
        return;
    }
    let block = m.div_ceil(rayon::current_num_threads()).max(64);
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

#[cfg(not(target_os = "macos"))]
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if c.is_empty() || k == 0 {
        return;
    }
    let block = m.div_ceil(rayon::current_num_threads()).max(64);
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
    if out.len() < parallel_min() {
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

fn reinsert_dim(t: Tensor, dim: usize, keepdim: bool) -> candle_core::Result<Tensor> {
    if keepdim { t.unsqueeze(dim) } else { Ok(t) }
}

// Above this, a dim-0 f32 sum becomes a ones-row matmul: much faster than
// candle's sequential sum, but it reassociates the additions (~1e-6 relative).
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
            COUNTERS.rowwise_calls.fetch_add(1, Ordering::Relaxed);
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

fn index_u32(index: &Tensor) -> candle_core::Result<Tensor> {
    index.contiguous()?.flatten_all()?.to_dtype(DType::U32)
}

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

fn scalar_leaf(graph: &Graph, leaves: &[u8], at: usize) -> Option<f32> {
    match &graph.nodes[at] {
        Node::Leaf {
            leaf,
            offset,
            shape,
            dtype,
        } if prod(shape) == 1 => {
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

fn cached_index(
    cache: &mut [Option<Tensor>],
    at: usize,
    index: &Tensor,
) -> candle_core::Result<Tensor> {
    if cache[at].is_none() {
        cache[at] = Some(index_u32(index)?);
        COUNTERS.index_builds.fetch_add(1, Ordering::Relaxed);
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
    let mut remaining = prep.consumers.clone();
    let mut outputs: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    let mut indices: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
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
                COUNTERS.gemm_calls.fetch_add(1, Ordering::Relaxed);
                let a = get(*a)?;
                let b = get(*b)?;
                let ar = a.rank();
                let br = b.rank();
                let m = a.dim(ar - 2)?;
                let k = a.dim(ar - 1)?;
                let n = b.dim(br - 1)?;
                let batch = candle_core::Shape::from_dims(&a.dims()[..ar - 2])
                    .broadcast_shape_binary_op(
                        &candle_core::Shape::from_dims(&b.dims()[..br - 2]),
                        "matmul",
                    )?;
                let mut a_shape = batch.dims().to_vec();
                a_shape.extend([m, k]);
                let mut b_shape = batch.dims().to_vec();
                b_shape.extend([k, n]);
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
        COUNTERS.candle_dispatches.fetch_add(1, Ordering::Relaxed);
        if let Some(started) = started {
            record(
                op_kind(node),
                started.elapsed().as_secs_f64(),
                prod(&prep.shapes[idx]),
            );
        }
        outputs[idx] = Some(out);
        for input in node_inputs(node) {
            release_input(&mut remaining, &prep.is_root, input, |i| {
                outputs[i] = None;
                indices[i] = None;
            });
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

// SAFETY: Readback owns its buffer; only the finalizer touches the pointer.
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

type ProfileRows = Vec<(String, f64, u64, u64)>;

static PROFILE: Mutex<Option<ProfileRows>> = Mutex::new(None);

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

fn evaluate(prep: &PreparedGraph, leaves: &[u8], seed: u32) -> Result<Readback> {
    let started = std::time::Instant::now();
    let result = evaluate_inner(prep, leaves, seed);
    COUNTERS.eval_ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

fn evaluate_inner(prep: &PreparedGraph, leaves: &[u8], seed: u32) -> Result<Readback> {
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
    let data = if flats.len() == 1 {
        flats.into_iter().next().unwrap()
    } else {
        Tensor::cat(&flats.iter().collect::<Vec<_>>(), 0).map_err(to_napi_err)?
    }
    .to_vec1::<f32>()
    .map_err(to_napi_err)?;
    Ok(vec_readback(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn eval_binary_scalar(name: &str, parameter: f64, x: f32, y: f32) -> f32 {
        let a = Tensor::new(x, &Device::Cpu).unwrap();
        let b = Tensor::new(y, &Device::Cpu).unwrap();
        eval_binary(name, parameter, &a, &b)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn eval_unary_scalar(name: &str, parameter: f64, x: f32) -> f32 {
        let a = Tensor::new(x, &Device::Cpu).unwrap();
        eval_unary(name, parameter, &a)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn assert_parity(scalar: f32, tensor: f32, what: &str) {
        let tol = 1e-5f32 * (1.0 + scalar.abs().max(tensor.abs()));
        let diff = (scalar - tensor).abs();
        assert!(
            diff <= tol,
            "{what}: scalar {scalar} vs tensor {tensor} (diff {diff})"
        );
    }

    #[test]
    fn binary_apply_matches_eval() {
        let cases = [
            (0.75f32, -1.25f32, 0.5f64),
            (2.0f32, 3.5f32, 2.0f64),
            (-0.5f32, 0.25f32, 0.5f64),
        ];
        for &name in Bin::all() {
            let op = Bin::parse(name).unwrap();
            for &(x, y, parameter) in &cases {
                let scalar = apply_bin(op, parameter as f32, x, y);
                let tensor = eval_binary_scalar(name, parameter, x, y);
                assert_parity(scalar, tensor, &format!("binary {name}({x}, {y}, {parameter})"));
            }
        }
    }

    #[test]
    fn unary_apply_matches_eval() {
        let default: &[(f32, f64)] = &[(0.75, 0.5), (-1.25, 2.0), (1.5, 2.0)];
        let positive: &[(f32, f64)] = &[(0.75, 0.5), (1.5, 2.0)];
        for &name in Un::all() {
            let op = Un::parse(name).unwrap();
            let cases = match name {
                "log" | "sqrt" => positive,
                _ => default,
            };
            for &(x, parameter) in cases {
                let scalar = apply_un(op, parameter as f32, x);
                let tensor = eval_unary_scalar(name, parameter, x);
                assert_parity(scalar, tensor, &format!("unary {name}({x}, {parameter})"));
            }
        }
    }
}