"""Streaming GPT-2 inference: layer-by-layer weight loading from SSD.

Implements the three I/O optimizations from the simulation layer as real code:
  1. Weight streaming — loads one layer at a time via numpy memmap, never holds
     the full model in RAM. Peak memory is O(one layer) not O(full model).
  2. Temporal cache — LRU cache for FFN down-projection rows. Consecutive tokens
     activate similar neurons (~97% overlap), so only ~3% delta is loaded per token.
  3. Sparse FFN — computes only active neurons in the down-projection. GELU
     produces ~90% zero activations, so 90% of the down-projection is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .weight_store import WeightStore


# ---------------------------------------------------------------------------
# Standalone math helpers
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Temporal Cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    data: np.ndarray
    indices: np.ndarray
    last_access: int = 0


class TemporalCache:
    """LRU cache for FFN down-projection neuron rows."""

    def __init__(self, budget_bytes: int, n_layers: int):
        self._budget = budget_bytes
        self._n_layers = n_layers
        self._entries: dict[int, _CacheEntry] = {}
        self._clock = 0
        self._total_hits = 0
        self._total_misses = 0
        self._bytes_loaded = 0

    def get_ffn_down_rows(
        self,
        store: WeightStore,
        layer_idx: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Return FFN down-projection rows for active neurons.

        Loads only the delta (rows not already cached) from disk.
        """
        self._clock += 1
        active_set = set(active_indices.tolist())

        if layer_idx in self._entries:
            entry = self._entries[layer_idx]
            cached_set = set(entry.indices.tolist())
            hits = active_set & cached_set
            misses = active_set - cached_set
            self._total_hits += len(hits)
            self._total_misses += len(misses)

            if not misses:
                entry.last_access = self._clock
                idx_map = {v: i for i, v in enumerate(entry.indices.tolist())}
                pick = np.array([idx_map[a] for a in active_indices.tolist()])
                return entry.data[pick].astype(np.float32)

            miss_arr = np.array(sorted(misses), dtype=np.intp)
            delta = store.load_rows(layer_idx, "ffn_down_weight", miss_arr)
            self._bytes_loaded += delta.nbytes

            new_indices = np.concatenate([entry.indices, miss_arr])
            new_data = np.concatenate([entry.data, delta])
            entry.indices = new_indices
            entry.data = new_data
            entry.last_access = self._clock

            idx_map = {v: i for i, v in enumerate(new_indices.tolist())}
            pick = np.array([idx_map[a] for a in active_indices.tolist()])
            return new_data[pick].astype(np.float32)

        full_rows = store.load_rows(layer_idx, "ffn_down_weight", active_indices)
        self._total_misses += len(active_indices)
        self._bytes_loaded += full_rows.nbytes
        self._entries[layer_idx] = _CacheEntry(
            data=np.array(full_rows),
            indices=np.array(active_indices, copy=True),
            last_access=self._clock,
        )
        self._evict()
        return full_rows.astype(np.float32)

    def _evict(self):
        total = sum(e.data.nbytes for e in self._entries.values())
        while total > self._budget and self._entries:
            oldest_key = min(self._entries, key=lambda k: self._entries[k].last_access)
            total -= self._entries[oldest_key].data.nbytes
            del self._entries[oldest_key]

    def stats(self) -> dict:
        total_bytes = sum(e.data.nbytes for e in self._entries.values())
        total_requests = self._total_hits + self._total_misses
        return {
            "hit_rate": self._total_hits / total_requests if total_requests else 0.0,
            "miss_rate": self._total_misses / total_requests if total_requests else 0.0,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "bytes_cached": total_bytes,
            "bytes_loaded_from_disk": self._bytes_loaded,
            "cache_utilization": total_bytes / self._budget if self._budget else 0.0,
        }

    def reset(self):
        self._entries.clear()
        self._clock = 0
        self._total_hits = 0
        self._total_misses = 0
        self._bytes_loaded = 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamingConfig:
    cache_budget_mb: int = 256
    sparse_threshold: float = 0.01
    use_sparse_ffn: bool = True


# ---------------------------------------------------------------------------
# Streaming GPT-2
# ---------------------------------------------------------------------------

class StreamingGPT2:
    """GPT-2 inference with layer streaming, temporal cache, and sparse FFN."""

    def __init__(
        self,
        store: WeightStore,
        config: StreamingConfig | None = None,
    ):
        self.store = store
        self.config = config or StreamingConfig()
        mc = store.config

        self.n_layers = mc["n_layers"]
        self.d_model = mc["d_model"]
        self.n_heads = mc["n_heads"]
        self.head_dim = mc["d_model"] // mc["n_heads"]
        self.ffn_dim = mc["ffn_dim"]
        self.vocab_size = mc["vocab_size"]

        self.embed = np.array(store.load_embed(), dtype=np.float32)
        self.pos_embed = np.array(store.load_pos_embed(), dtype=np.float32)
        ln_f_w, ln_f_b = store.load_final_ln()
        self.ln_f_w = np.array(ln_f_w, dtype=np.float32)
        self.ln_f_b = np.array(ln_f_b, dtype=np.float32)

        self.kv_cache: list[tuple[np.ndarray, np.ndarray] | None] = [None] * self.n_layers
        self._seq_len = 0

        self.temporal_cache = TemporalCache(
            budget_bytes=self.config.cache_budget_mb * 1024 ** 2,
            n_layers=self.n_layers,
        )

    # ----- Public interface -----

    def forward(
        self,
        token_ids: np.ndarray,
        start_pos: int | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """Run forward pass. Returns (logits, hidden_states_or_None)."""
        if start_pos is None:
            start_pos = self._seq_len

        seq_len = token_ids.shape[-1] if token_ids.ndim > 0 else 1
        token_ids = token_ids.reshape(-1)

        x = self.embed[token_ids] + self.pos_embed[start_pos:start_pos + seq_len]
        if x.ndim == 1:
            x = x.reshape(1, -1)

        hidden_states = [x.copy()] if output_hidden_states else None

        for i in range(self.n_layers):
            weights = self.store.load_layer(i)
            w = {k: np.array(v, dtype=np.float32) if v.dtype != np.float32 else np.array(v)
                 for k, v in weights.items()
                 if k.startswith("ln_") or k.startswith("attn_")}
            x = self._block(x, weights, w, i)
            if output_hidden_states:
                hidden_states.append(x.copy())
            del weights, w

        x = _layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = x @ self.embed.T

        self._seq_len = start_pos + seq_len
        return logits, hidden_states

    def __call__(self, input_ids, attention_mask=None,
                 output_hidden_states=False, **kwargs):
        """Numpy-native interface. Returns StreamingOutput."""
        if hasattr(input_ids, 'numpy'):
            ids = input_ids.squeeze(0).numpy()
        else:
            ids = np.asarray(input_ids).reshape(-1)

        logits, hidden = self.forward(
            ids,
            output_hidden_states=output_hidden_states,
        )
        return _StreamingOutput(
            logits=logits[np.newaxis, ...],
            hidden_states=tuple(h[np.newaxis, ...] for h in hidden) if hidden else None,
        )

    def generate(self, prompt_tokens: np.ndarray, max_new_tokens: int,
                 temperature: float = 1.0) -> list[int]:
        self.reset()
        prompt = np.asarray(prompt_tokens, dtype=np.int64)
        logits, _ = self.forward(prompt, start_pos=0)

        generated = []
        for _ in range(max_new_tokens):
            next_logits = logits[-1] / temperature
            probs = _softmax(next_logits)
            token = int(np.argmax(probs))
            generated.append(token)
            logits, _ = self.forward(np.array([token], dtype=np.int64))

        return generated

    def reset(self):
        self.kv_cache = [None] * self.n_layers
        self._seq_len = 0
        self.temporal_cache.reset()

    def memory_stats(self) -> dict:
        embed_mb = self.embed.nbytes / 1024 ** 2
        pos_mb = self.pos_embed.nbytes / 1024 ** 2
        kv_bytes = sum(
            k.nbytes + v.nbytes
            for kv in self.kv_cache if kv is not None
            for k, v in [kv]
        )
        tc = self.temporal_cache.stats()
        return {
            "embed_mb": round(embed_mb, 1),
            "pos_embed_mb": round(pos_mb, 1),
            "kv_cache_mb": round(kv_bytes / 1024 ** 2, 1),
            "temporal_cache_mb": round(tc["bytes_cached"] / 1024 ** 2, 1),
            "total_mb": round(
                embed_mb + pos_mb + kv_bytes / 1024 ** 2
                + tc["bytes_cached"] / 1024 ** 2, 1
            ),
        }

    # ----- Internal -----

    def _block(self, x, raw_weights, float_weights, layer_idx):
        normed = _layer_norm(x, float_weights["ln_1_weight"], float_weights["ln_1_bias"])
        attn_out = self._attention(normed, float_weights, layer_idx)
        x = x + attn_out

        normed = _layer_norm(x, float_weights["ln_2_weight"], float_weights["ln_2_bias"])
        if self.config.use_sparse_ffn:
            ffn_out = self._sparse_ffn(normed, raw_weights, layer_idx)
        else:
            ffn_out = self._dense_ffn(normed, raw_weights)
        x = x + ffn_out
        return x

    def _attention(self, x, w, layer_idx):
        seq_len = x.shape[0]
        qkv_w = w.get("attn_qkv_weight")
        if qkv_w is None:
            qkv_w = np.array(self.store.load_layer(layer_idx)["attn_qkv_weight"], dtype=np.float32)
        qkv_b = w.get("attn_qkv_bias")
        if qkv_b is None:
            qkv_b = np.array(self.store.load_layer(layer_idx)["attn_qkv_bias"], dtype=np.float32)
        proj_w = w.get("attn_proj_weight")
        if proj_w is None:
            proj_w = np.array(self.store.load_layer(layer_idx)["attn_proj_weight"], dtype=np.float32)
        proj_b = w.get("attn_proj_bias")
        if proj_b is None:
            proj_b = np.array(self.store.load_layer(layer_idx)["attn_proj_bias"], dtype=np.float32)

        qkv = x @ qkv_w + qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)

        q = q.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)

        if self.kv_cache[layer_idx] is not None:
            prev_k, prev_v = self.kv_cache[layer_idx]
            k = np.concatenate([prev_k, k], axis=1)
            v = np.concatenate([prev_v, v], axis=1)
        self.kv_cache[layer_idx] = (k, v)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 2, 1)) * scale

        kv_len = k.shape[1]
        if seq_len > 1:
            mask = np.triu(np.ones((seq_len, kv_len), dtype=np.float32), k=kv_len - seq_len + 1)
            scores = scores + mask * (-1e9)

        attn = _softmax(scores, axis=-1)
        out = attn @ v

        out = out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return out @ proj_w + proj_b

    def _dense_ffn(self, x, raw_weights):
        up_w = np.array(raw_weights["ffn_up_weight"], dtype=np.float32)
        up_b = np.array(raw_weights["ffn_up_bias"], dtype=np.float32)
        down_w = np.array(raw_weights["ffn_down_weight"], dtype=np.float32)
        down_b = np.array(raw_weights["ffn_down_bias"], dtype=np.float32)

        hidden = _gelu(x @ up_w + up_b)
        return hidden @ down_w + down_b

    def _sparse_ffn(self, x, raw_weights, layer_idx):
        up_w = np.array(raw_weights["ffn_up_weight"], dtype=np.float32)
        up_b = np.array(raw_weights["ffn_up_bias"], dtype=np.float32)
        down_b = np.array(raw_weights["ffn_down_bias"], dtype=np.float32)

        hidden = _gelu(x @ up_w + up_b)

        seq_len = x.shape[0] if x.ndim == 2 else 1
        if seq_len > 1:
            down_w = np.array(raw_weights["ffn_down_weight"], dtype=np.float32)
            return hidden @ down_w + down_b

        active_mask = np.abs(hidden.ravel()) > self.config.sparse_threshold
        active_idx = np.nonzero(active_mask)[0]

        if len(active_idx) == 0 or len(active_idx) == self.ffn_dim:
            down_w = np.array(raw_weights["ffn_down_weight"], dtype=np.float32)
            return hidden @ down_w + down_b

        active_down_w = self.temporal_cache.get_ffn_down_rows(
            self.store, layer_idx, active_idx,
        )

        output = hidden[..., active_idx] @ active_down_w + down_b
        return output


# ---------------------------------------------------------------------------
# Output container + Adapter for engine.py
# ---------------------------------------------------------------------------

class _StreamingOutput:
    __slots__ = ("logits", "hidden_states")

    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class StreamingModelAdapter:
    """Wraps StreamingGPT2 to return torch tensors for SpeculativeEngine."""

    def __init__(self, model: StreamingGPT2):
        self._model = model

    def __call__(self, input_ids=None, attention_mask=None,
                 output_hidden_states=False, **kwargs):
        import torch

        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.squeeze(0).cpu().numpy()
        else:
            ids = np.asarray(input_ids).reshape(-1)

        logits, hidden = self._model.forward(
            ids, output_hidden_states=output_hidden_states,
        )

        logits_t = torch.from_numpy(logits[np.newaxis, ...].copy()).float()

        hidden_t = None
        if hidden is not None:
            hidden_t = tuple(
                torch.from_numpy(h[np.newaxis, ...].copy()).float()
                for h in hidden
            )

        return _AdapterOutput(logits=logits_t, hidden_states=hidden_t)

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    def to(self, device):
        return self


class _AdapterOutput:
    __slots__ = ("logits", "hidden_states")

    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states
