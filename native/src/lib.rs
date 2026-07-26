use candle_core::{DType, Device, Tensor};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;
use std::sync::OnceLock;

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
}

fn prod(shape: &[usize]) -> usize {
    shape.iter().product()
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

fn run_graph(graph: &Graph, leaves: &[f32]) -> candle_core::Result<Tensor> {
    let mut outputs: Vec<Tensor> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let get = |i: usize| -> candle_core::Result<&Tensor> {
            outputs.get(i).ok_or_else(|| {
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
                Tensor::from_vec(data.to_vec(), shape.clone(), device())?
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
        outputs.push(out);
    }
    outputs.pop().ok_or_else(|| {
        candle_core::Error::Msg("empty graph: no root node".to_string())
    })
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
    let graph: Graph = serde_json::from_str(&graph_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("invalid graph JSON: {e}")))?;
    let output = run_graph(&graph, &leaves).map_err(to_napi_err)?;
    output.device().synchronize().map_err(to_napi_err)?;
    let flat = output
        .contiguous()
        .and_then(|t| t.flatten_all())
        .map_err(to_napi_err)?;
    let data = flat.to_vec1::<f32>().map_err(to_napi_err)?;
    Ok(vec_readback(data))
}
