"""Shared shape/config data for the PyTorch bench mirrors (PLAN-V2 §4.2, W0.4).

Numbers here are copied verbatim from `bench/lib/sizes.ts` — the TypeScript
side is the single source of truth; if a shape or config changes there, it
must change here too, in the same commit. Do not invent a shape that does
not already have a `bench/lib/sizes.ts` counterpart.

Every `*_FULL` config exists for documentation and for a future `--full`
run; Phase A (PLAN-V2 §5A.0) never runs `--full`, so every bench_*.py
script here defaults to the `*_SMOKE` config, exactly like its TypeScript
counterpart defaults to smoke unless given `--full`.
"""

# --- mlp (bench/lib/sizes.ts: MLP_LEGACY / MLP_SMOKE) -----------------------

MLP_LEGACY = [
    {"id": "mlp-legacy-b64", "inputDim": 784, "hiddenDim": 256, "outputDim": 10, "batch": 64},
    {"id": "mlp-legacy-b512", "inputDim": 784, "hiddenDim": 256, "outputDim": 10, "batch": 512},
]

MLP_SMOKE = [
    {"id": "mlp-legacy-smoke-b8", "inputDim": 784, "hiddenDim": 256, "outputDim": 10, "batch": 8},
]

# --- nanoGPT (bench/lib/sizes.ts: NANOGPT_SIZES / NANOGPT_SMOKE) -----------
#
# S  B32 T64  d128 H4 L4 V65
# M  B32 T128 d256 H8 L6 V65
# L  B64 T256 d384 H6 L6 V65

NANOGPT_SIZES = [
    {"id": "nanogpt-s", "batch": 32, "blockSize": 64, "nEmbd": 128, "nHead": 4, "nLayer": 4, "vocabSize": 65},
    {"id": "nanogpt-m", "batch": 32, "blockSize": 128, "nEmbd": 256, "nHead": 8, "nLayer": 6, "vocabSize": 65},
    {"id": "nanogpt-l", "batch": 64, "blockSize": 256, "nEmbd": 384, "nHead": 6, "nLayer": 6, "vocabSize": 65},
]

NANOGPT_SMOKE = {
    "id": "nanogpt-smoke",
    "batch": 4,
    "blockSize": 16,
    "nEmbd": 32,
    "nHead": 4,
    "nLayer": 1,
    "vocabSize": 65,
}

# --- micro ops (bench/lib/sizes.ts: MATMUL_*, ELEMENTWISE_*, REDUCE_*,
#     SOFTMAX_LN_*) — `bench_ops.py`'s cases. --------------------------------

MATMUL_FULL = {"square": [256, 512, 1024, 2048]}
MATMUL_SMOKE = {"square": [64, 128]}

ELEMENTWISE_FULL = {"sizes": [4_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000], "chainLengths": [1, 5, 12, 24]}
ELEMENTWISE_SMOKE = {"sizes": [4_000, 16_000], "chainLengths": [1, 5]}

REDUCE_SHAPES_FULL = [
    {"label": "small", "shape": [1024, 64]},
    {"label": "below-route", "shape": [2048, 256]},
    {"label": "gemv-route", "shape": [8192, 256]},
    {"label": "large", "shape": [65536, 128]},
]
REDUCE_SHAPES_SMOKE = [
    {"label": "small", "shape": [128, 32]},
    {"label": "below-route", "shape": [256, 64]},
]

SOFTMAX_LN_FULL = {"rows": 4096, "widths": [64, 128, 384, 1024], "ceRows": 16384, "ceCols": 65}
SOFTMAX_LN_SMOKE = {"rows": 256, "widths": [64, 128], "ceRows": 512, "ceCols": 65}
