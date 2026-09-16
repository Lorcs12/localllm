# Adaptive-EAGLE: SMW-based Online Speculative Decoding

## What This Is

A self-adapting draft head for speculative decoding that replaces gradient-based online adaptation with trace-bounded Sherman-Morrison-Woodbury (SMW) rank-1 updates. Target-model rejections become free training signals absorbed in O(D^2) — no gradients, no optimizer state, sub-millisecond per update.

## Architecture

Five-phase pipeline per generation round:

1. **CUP Prediction** (~2ms) — predict which cold neurons will fire across the draft window
2. **Concurrent Draft + Fetch** (~80ms) — SMW Ridge head drafts K tokens while io_uring fetches slab-packed weights from SSD via O_DIRECT (bypasses L3 cache)
3. **Sparse Verification** (~20ms) — fused gather-GEMM on active neurons + 4-bit quantized KV cache attention
4. **SMW Update** (~0.22ms) — trace-bounded covariance update + sparse margin (2 rows of W)
5. **Loop Reset** — resume drafting with adapted head

## Module Map

| File | Role |
|------|------|
| `config.py` | `OSDConfig` frozen dataclass — all hyperparameters |
| `draft_head.py` | **Mathematical core.** `SMWDraftHead` with trace-bounded SMW + sparse margin updates |
| `features.py` | Last-token hidden state extraction from HuggingFace models |
| `engine.py` | `SpeculativeEngine` — draft/verify/update loop orchestration |
| `metrics.py` | `AcceptanceTracker` — per-round acceptance rates, adaptation gain |
| `neuron_map.py` | Hot/cold neuron segregation + slab-packed weight index |
| `predictor.py` | CUP (Co-activation Unit Predictor) — 2-layer MLP for neuron activation |
| `slab_fetch.py` | Async slab fetcher with simulated SSD timing |
| `hardware.py` | L3 cache simulator, bandwidth models |
| `sparse_ops.py` | Fused gather-GEMM reference (no-copy sparse matmul) |
| `kv_cache.py` | INT4 quantization + H2O/StreamingLLM heavy-hitter eviction |
| `hybrid_mamba.py` | Hybrid Mamba architecture simulation: Transformer vs Mamba at million-token scale |
| `kv_router.py` | KV Router: hot/cold KV page management with async SSD fetch for infinite context |
| `layer_stream.py` | Layer-streaming inference: static vs 1-buffer vs ping-pong 2-buffer strategies for RAM reduction |
| `hybrid_engine.py` | Hybrid CPU+GPU inference: PowerInfer-style sparse/dense split with DirectStorage + AVX-512 |
| `inference_engine.py` | Inference engine tier comparison: PyTorch vs GGML+ZigZag vs DirectStorage for 70B+ models on consumer hardware |
| `pipeline.py` | Full 5-phase pipeline simulator with timing breakdown |

## Key Math (draft_head.py)

**Cold start:** `W = W_lm_head`, `Ainv = (1/lambda)*I`, `XtWX = lambda*I`
The head IS the original LM head at token 0.

**Per-rejection update (O(D^2)):**
```
# Phase 1: Trace-bounded covariance
Ainv *= 1/lambda_forget          # decay old curvature
XtWX *= lambda_forget
Ainv -= (w/denom) * outer(Ainv@x, Ainv@x)   # SMW rank-1 update
if trace(Ainv) > max_trace: rescale

# Phase 2: Sparse margin (O(D))
W[:, target]  += x_adj           # boost correct token
W[:, drafted] -= x_adj           # penalize mistake
```

**Sliding window:** FIFO of last 128 corrections. Evicts oldest via rank-1 downdate.

**Periodic reinversion:** Every 200 updates, `Ainv = inv(XtWX)` to bound float drift.

## Dependencies

- `numpy` — all linear algebra
- `torch` + `transformers` — model loading and feature extraction (engine.py, features.py only)
- `neuralbyte.ridge` — `invert()` for periodic reinversion

## Running Tests

```bash
pytest tests/test_spec_decode.py           # Pure math, no models needed
pytest tests/test_spec_decode_io.py        # Neuron map, CUP, slab fetch
pytest tests/test_spec_decode_hardware.py  # L3 cache, sparse ops, KV cache, pipeline
```

## Conventions

- All math is float64 numpy. No PyTorch in the adaptation loop.
- `SMWDraftHead` is a mutable dataclass — `W`, `Ainv`, `XtWX` are updated in-place.
- Feature extraction uses `torch.inference_mode()` and single-layer hidden states.
- Simulation modules use deterministic timing models, not real hardware.
