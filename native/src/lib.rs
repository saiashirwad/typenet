use candle_core::{DType, Device, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
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
}

#[derive(Debug, Deserialize)]
struct Graph {
    nodes: Vec<Node>,
    /// Indices of the output nodes. Defaults to the last node, so
    /// single-root graphs from older callers keep working.
    #[serde(default)]
    roots: Option<Vec<usize>>,
    /// Device hint from the JS side: "cpu" pins evaluation to the
    /// plan/exec CPU evaluator (tiny graphs, where per-kernel dispatch
    /// dwarfs the compute); anything else uses the best available device.
    /// Read only via the JSON suffix check in eval_graph, not after
    /// deserialization.
    #[serde(default)]
    #[allow(dead_code)]
    device: Option<String>,
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
        };
        shapes.push(shape);
    }
    Ok(shapes)
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
        "negDiv" => elementwise(a, b, |x, y| x.neg()? / y),
        "halfDiv" => elementwise(a, b, |x, y| (x * 0.5)? / y),
        "mulSign" => elementwise(a, b, |x, y| {
            // sign(y) = (y > 0) - (y < 0)
            let zeros = y.zeros_like()?;
            let pos = mask_f32(&y.gt(&zeros)?)?;
            let neg = mask_f32(&y.lt(&zeros)?)?;
            x * &(pos - neg)?
        }),
        "reluGrad" => elementwise(a, b, |x, y| {
            x * &mask_f32(&y.gt(&y.zeros_like()?)?)?
        }),
        "leakyReluGrad" => elementwise(a, b, |x, y| {
            // where y > 0: x, else parameter * x  ==  x * (m + (1-m)*p)
            let m = mask_f32(&y.gt(&y.zeros_like()?)?)?;
            let weights = (&m + &(m.ones_like()? - &m)? * parameter)?;
            x * &weights
        }),
        "sigmoidGrad" => elementwise(a, b, |x, y| (x * y)? * &(y.ones_like()? - y)?),
        "tanhGrad" => elementwise(a, b, |x, y| x * &(y.ones_like()? - &y.sqr()?)?),
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
        "leakyRelu" => {
            // max(x, 0) + parameter * min(x, 0)
            let zeros = a.zeros_like()?;
            let pos = a.maximum(&zeros)?;
            let neg = a.minimum(&zeros)?;
            pos + (neg * parameter)?
        }
        "sigmoid" => {
            // 1 / (1 + exp(-x))
            let ones = a.ones_like()?;
            &ones / &(&ones + &a.neg()?.exp()?)?
        }
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

fn scalar_binary(kind: &str, parameter: f64, a: f32, b: f32) -> candle_core::Result<f32> {
    let p = parameter as f32;
    Ok(match kind {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        "negDiv" => -a / b,
        "halfDiv" => 0.5 * a / b,
        "mulSign" => a * ((b > 0.0) as u8 as f32 - (b < 0.0) as u8 as f32),
        "reluGrad" => {
            if b > 0.0 {
                a
            } else {
                0.0
            }
        }
        "leakyReluGrad" => a * if b > 0.0 { 1.0 } else { p },
        "sigmoidGrad" => a * b * (1.0 - b),
        "tanhGrad" => a * (1.0 - b * b),
        other => {
            return Err(candle_core::Error::Msg(format!(
                "unknown binary op: {other}"
            )))
        }
    })
}

fn scalar_unary(kind: &str, parameter: f64, x: f32) -> candle_core::Result<f32> {
    let p = parameter as f32;
    Ok(match kind {
        "pow" => x.powf(p),
        "neg" => -x,
        "exp" => x.exp(),
        "log" => x.ln(),
        "sqrt" => x.sqrt(),
        "abs" => x.abs(),
        "relu" => x.max(0.0),
        "leakyRelu" => {
            if x > 0.0 {
                x
            } else {
                p * x
            }
        }
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        "scalePowGrad" => p * x.powf(p - 1.0),
        other => {
            return Err(candle_core::Error::Msg(format!(
                "unknown unary op: {other}"
            )))
        }
    })
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

struct MemberPlan {
    /// Node index in the graph (op kind/parameter read from there).
    node: usize,
    /// Shape this member's pass produces (the group output shape for
    /// main members; the member's own smaller shape for small members).
    out_shape: Vec<usize>,
    /// Fully resolved inputs (1 for unary, 2 for binary).
    inputs: Vec<ChildRef>,
}

struct GroupPlan {
    leader: usize,
    out_shape: Vec<usize>,
    /// Members smaller than the output shape, topo order; temp index =
    /// position. Their inputs can only be Buffer or earlier Temps.
    small_members: Vec<MemberPlan>,
    /// Same-shape-as-output members, topo order; scratch slot = position;
    /// the leader is last.
    main_members: Vec<MemberPlan>,
}

/// Standalone (ungrouped) elementwise node: resolved Buffer inputs.
struct EwisePlan {
    out_shape: Vec<usize>,
    inputs: Vec<ChildRef>,
}

struct PreparedGraph {
    graph: Graph,
    shapes: Vec<Vec<usize>>,
    roots: Vec<usize>,
    live: Vec<bool>,
    /// group index per member node (skip during the main loop); leaders
    /// trigger execution.
    group_of: Vec<Option<usize>>,
    groups: Vec<GroupPlan>,
    /// Per-node plans for standalone elementwise nodes.
    ewise: Vec<Option<EwisePlan>>,
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
                    let inputs = inputs
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
                        node: m,
                        out_shape: out_shape.clone(),
                        inputs,
                    });
                } else {
                    // Small members can only read buffers or earlier temps.
                    temp_of[m] = Some(small_members.len());
                    let target = shapes[m].clone();
                    let inputs = inputs
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
                        node: m,
                        out_shape: target,
                        inputs,
                    });
                }
            }
            groups.push(GroupPlan {
                leader,
                out_shape,
                small_members,
                main_members,
            });
        }

        let mut ewise: Vec<Option<EwisePlan>> = (0..n).map(|_| None).collect();
        for (idx, node) in graph.nodes.iter().enumerate() {
            if !live[idx] || fusion.group_of[idx].is_some() || !is_elementwise(node) {
                continue;
            }
            let target = shapes[idx].clone();
            ewise[idx] = Some(EwisePlan {
                inputs: node_inputs(node)
                    .iter()
                    .map(|&c| buffer_child(c, &target))
                    .collect(),
                out_shape: target,
            });
        }

        Ok(PreparedGraph {
            graph,
            shapes,
            roots,
            live,
            group_of: fusion.group_of,
            groups,
            ewise,
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

/// One elementwise member (op from `graph.nodes[mp.node]`) as a single
/// pass over `target_shape`.
fn exec_member(
    node: &Node,
    inputs: &[ChildRef],
    target_shape: &[usize],
    buffers: &[Option<Vec<f32>>],
    temps: &[Vec<f32>],
    coords: &mut [usize],
) -> candle_core::Result<Vec<f32>> {
    let n = prod(target_shape);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, target_shape, coords);
        let value = match node {
            Node::Binary {
                kind, parameter, ..
            } => scalar_binary(
                kind,
                *parameter,
                read_ref(&inputs[0], i, coords, buffers, temps, &[]),
                read_ref(&inputs[1], i, coords, buffers, temps, &[]),
            )?,
            Node::Unary {
                kind, parameter, ..
            } => scalar_unary(
                kind,
                *parameter,
                read_ref(&inputs[0], i, coords, buffers, temps, &[]),
            )?,
            _ => unreachable!("elementwise plans only contain elementwise nodes"),
        };
        out.push(value);
    }
    Ok(out)
}

fn exec_group(
    plan: &GroupPlan,
    graph: &Graph,
    buffers: &[Option<Vec<f32>>],
) -> candle_core::Result<Vec<f32>> {
    let mut temps: Vec<Vec<f32>> = Vec::with_capacity(plan.small_members.len());
    let mut coords = vec![0usize; plan.out_shape.len()];
    for sm in &plan.small_members {
        temps.push(exec_member(
            &graph.nodes[sm.node],
            &sm.inputs,
            &sm.out_shape,
            buffers,
            &temps,
            &mut coords,
        )?);
    }
    let n = prod(&plan.out_shape);
    let mut scratch = vec![0f32; plan.main_members.len()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, &plan.out_shape, &mut coords);
        for (slot, mm) in plan.main_members.iter().enumerate() {
            let value = match &graph.nodes[mm.node] {
                Node::Binary {
                    kind, parameter, ..
                } => scalar_binary(
                    kind,
                    *parameter,
                    read_ref(&mm.inputs[0], i, &coords, buffers, &temps, &scratch),
                    read_ref(&mm.inputs[1], i, &coords, buffers, &temps, &scratch),
                )?,
                Node::Unary {
                    kind, parameter, ..
                } => scalar_unary(
                    kind,
                    *parameter,
                    read_ref(&mm.inputs[0], i, &coords, buffers, &temps, &scratch),
                )?,
                _ => unreachable!("fusion groups only contain elementwise nodes"),
            };
            scratch[slot] = value;
        }
        out.push(scratch[plan.main_members.len() - 1]);
    }
    Ok(out)
}

/// Whole-graph execution from a prepared plan: leaf copies + raw loops,
/// no parsing or planning. Returns all roots concatenated.
fn execute(prep: &PreparedGraph, leaves: &[f32]) -> candle_core::Result<Vec<f32>> {
    let graph = &prep.graph;
    let n = graph.nodes.len();
    let mut buffers: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    let mut coords: Vec<usize> = Vec::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !prep.live[idx] {
            continue;
        }
        if let Some(g) = prep.group_of[idx] {
            if prep.groups[g].leader == idx {
                buffers[idx] = Some(exec_group(&prep.groups[g], graph, &buffers)?);
            }
            continue;
        }
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
                let plan = prep.ewise[idx].as_ref().unwrap();
                coords.resize(plan.out_shape.len(), 0);
                exec_member(node, &plan.inputs, &plan.out_shape, &buffers, &[], &mut coords)?
            }
            Node::Matmul { a, b } => {
                tiny_matmul(get(*a)?, &prep.shapes[*a], get(*b)?, &prep.shapes[*b])?
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
        };
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

/// Naive matmul with typenet's batch-dim broadcasting, on Vecs. Only used
/// for tiny graphs, where this beats one candle dispatch.
fn tiny_matmul(
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
        for i in 0..m {
            for l in 0..k {
                let av = adata[ao + i * k + l];
                for j in 0..n {
                    out[oo + i * n + j] += av * bdata[bo + l * n + j];
                }
            }
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
    let mut coords = vec![0usize; shape.len()];
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        // Output element i -> input coords with the reduced coord at 0.
        let mut rem = i;
        for j in (0..shape.len()).rev() {
            let size = if j == dim { 1 } else { shape[j] };
            coords[j] = rem % size;
            rem /= size;
        }
        let mut base = 0usize;
        for j in 0..shape.len() {
            base += coords[j] * strides[j];
        }
        match kind {
            "sum" => {
                let mut acc = 0f32;
                for dd in 0..d {
                    acc += data[base + dd * step];
                }
                out.push(acc);
            }
            "max" => {
                let mut acc = data[base];
                for dd in 1..d {
                    let v = data[base + dd * step];
                    if v > acc {
                        acc = v;
                    }
                }
                out.push(acc);
            }
            // First index wins on ties, matching the CPU kernel.
            "argmax" => {
                let mut best = 0usize;
                let mut acc = data[base];
                for dd in 1..d {
                    let v = data[base + dd * step];
                    if v > acc {
                        acc = v;
                        best = dd;
                    }
                }
                out.push(best as f32);
            }
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown reduce op: {other}"
                )))
            }
        }
    }
    Ok(out)
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
    let n = prod(to);
    let mut coords = vec![0usize; to.len()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, to, &mut coords);
        out.push(read_bcast(data, &strides, same, i, &coords));
    }
    out
}

fn tiny_permute(data: &[f32], shape: &[usize], order: &[usize]) -> Vec<f32> {
    let in_strides = row_major_strides(shape);
    let out_shape: Vec<usize> = order.iter().map(|&d| shape[d]).collect();
    // Output coords -> input index via the permuted input strides.
    let strides: Vec<usize> = order.iter().map(|&d| in_strides[d]).collect();
    let n = prod(&out_shape);
    let mut coords = vec![0usize; out_shape.len()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, &out_shape, &mut coords);
        let mut idx = 0usize;
        for j in 0..coords.len() {
            idx += coords[j] * strides[j];
        }
        out.push(data[idx]);
    }
    out
}

fn tiny_narrow(data: &[f32], shape: &[usize], dim: usize, start: usize, length: usize) -> Vec<f32> {
    let strides = row_major_strides(shape);
    let mut out_shape = shape.to_vec();
    out_shape[dim] = length;
    let n = prod(&out_shape);
    let mut coords = vec![0usize; shape.len()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, &out_shape, &mut coords);
        let mut idx = start * strides[dim];
        for j in 0..shape.len() {
            idx += coords[j] * strides[j];
        }
        out.push(data[idx]);
    }
    out
}

fn tiny_cat(
    adata: &[f32],
    ashape: &[usize],
    bdata: &[f32],
    bshape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let astr = row_major_strides(ashape);
    let bstr = row_major_strides(bshape);
    let mut out_shape = ashape.to_vec();
    out_shape[dim] += bshape[dim];
    let n = prod(&out_shape);
    let mut coords = vec![0usize; out_shape.len()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        flat_to_coords(i, &out_shape, &mut coords);
        let value = if coords[dim] < ashape[dim] {
            let mut idx = 0usize;
            for j in 0..coords.len() {
                idx += coords[j] * astr[j];
            }
            adata[idx]
        } else {
            coords[dim] -= ashape[dim];
            let mut idx = 0usize;
            for j in 0..coords.len() {
                idx += coords[j] * bstr[j];
            }
            bdata[idx]
        };
        out.push(value);
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

fn run_graph(
    graph: &Graph,
    leaves: &[f32],
    device: &Device,
) -> candle_core::Result<Vec<Tensor>> {
    let roots: Vec<usize> = match &graph.roots {
        Some(roots) => roots.clone(),
        None => vec![graph.nodes.len().saturating_sub(1)],
    };

    // Dead-code elimination: mark everything reachable from the roots and
    // never touch the rest (dead leaves aren't even copied to the device).
    let n = graph.nodes.len();
    let mut live = vec![false; n];
    let mut stack = roots.clone();
    while let Some(i) = stack.pop() {
        if live[i] {
            continue;
        }
        live[i] = true;
        stack.extend(node_inputs(&graph.nodes[i]));
    }

    let mut outputs: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !live[idx] {
            continue;
        }
        let get = |i: usize| -> candle_core::Result<&Tensor> {
            outputs.get(i).and_then(|t| t.as_ref()).ok_or_else(|| {
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
            } => eval_binary(kind, *parameter, get(*a)?, get(*b)?)?,
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
                let a = a.broadcast_as(a_shape)?.contiguous()?;
                let b = b.broadcast_as(b_shape)?.contiguous()?;
                a.matmul(&b)?
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
        };
        outputs[idx] = Some(out);
    }
    roots
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
pub fn eval_graph(graph_json: String, leaves: Float32Array) -> Result<Readback> {
    // Tiny CPU-pinned graphs bypass candle entirely (plan/exec evaluator).
    // serializeLazyGraph emits `device` last, so the suffix check avoids a
    // full JSON parse on the hot path (plan-cache hits).
    if graph_json.ends_with("\"device\":\"cpu\"}") {
        let prep = prepared(&graph_json)?;
        let data = execute(&prep, &leaves).map_err(to_napi_err)?;
        return Ok(vec_readback(data));
    }
    let graph: Graph = serde_json::from_str(&graph_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("invalid graph JSON: {e}")))?;
    let device = device();
    let outputs = run_graph(&graph, &leaves, device).map_err(to_napi_err)?;
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
