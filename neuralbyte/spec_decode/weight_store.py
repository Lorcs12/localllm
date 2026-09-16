"""Weight store: export HuggingFace models to per-layer numpy files.

Provides memmap-based loading so that weights are never fully resident in RAM.
One layer at a time is loaded, used for a forward pass, and freed — total RAM
usage is O(one layer) instead of O(full model).

Export layout:
    store_dir/
      config.json
      _EXPORT_COMPLETE
      embed.npy            [vocab, d_model] FP16
      pos_embed.npy        [max_pos, d_model] FP16
      ln_f_weight.npy      [d_model] FP32
      ln_f_bias.npy        [d_model] FP32
      layers/
        0/
          ln_1_weight.npy, ln_1_bias.npy
          attn_qkv_weight.npy, attn_qkv_bias.npy
          attn_proj_weight.npy, attn_proj_bias.npy
          ln_2_weight.npy, ln_2_bias.npy
          ffn_up_weight.npy, ffn_up_bias.npy
          ffn_down_weight.npy, ffn_down_bias.npy
        1/ ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_SENTINEL = "_EXPORT_COMPLETE"

_LAYER_FILES = [
    "ln_1_weight.npy", "ln_1_bias.npy",
    "attn_qkv_weight.npy", "attn_qkv_bias.npy",
    "attn_proj_weight.npy", "attn_proj_bias.npy",
    "ln_2_weight.npy", "ln_2_bias.npy",
    "ffn_up_weight.npy", "ffn_up_bias.npy",
    "ffn_down_weight.npy", "ffn_down_bias.npy",
]

_WEIGHT_MAP = {
    "ln_1_weight": ("ln_1.weight", np.float32),
    "ln_1_bias": ("ln_1.bias", np.float32),
    "attn_qkv_weight": ("attn.c_attn.weight", np.float16),
    "attn_qkv_bias": ("attn.c_attn.bias", np.float16),
    "attn_proj_weight": ("attn.c_proj.weight", np.float16),
    "attn_proj_bias": ("attn.c_proj.bias", np.float16),
    "ln_2_weight": ("ln_2.weight", np.float32),
    "ln_2_bias": ("ln_2.bias", np.float32),
    "ffn_up_weight": ("mlp.c_fc.weight", np.float16),
    "ffn_up_bias": ("mlp.c_fc.bias", np.float16),
    "ffn_down_weight": ("mlp.c_proj.weight", np.float16),
    "ffn_down_bias": ("mlp.c_proj.bias", np.float16),
}


class WeightStore:
    """Per-layer weight storage with memmap loading."""

    def __init__(self, store_dir: str | Path):
        self._dir = Path(store_dir)
        if not (self._dir / _SENTINEL).exists():
            raise ValueError(
                f"Incomplete or missing export at {self._dir}. "
                "Run WeightStore.export() first."
            )
        with open(self._dir / "config.json") as f:
            self._config = json.load(f)

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def n_layers(self) -> int:
        return self._config["n_layers"]

    @staticmethod
    def export(model_name: str, output_dir: str | Path) -> None:
        """Export a HuggingFace GPT-2 model to streaming format.

        Loads with dtype=float16 and low_cpu_mem_usage to minimize RAM.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        hf_config = AutoConfig.from_pretrained(model_name)
        config = {
            "n_layers": hf_config.n_layer,
            "d_model": hf_config.n_embd,
            "n_heads": hf_config.n_head,
            "ffn_dim": 4 * hf_config.n_embd,
            "vocab_size": hf_config.vocab_size,
            "max_position": hf_config.n_positions,
            "model_name": model_name,
        }
        with open(output / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, low_cpu_mem_usage=True,
        )

        sd = model.state_dict()

        def _save(name: str, key: str, dtype=np.float16):
            arr = sd[key].float().numpy().astype(dtype)
            np.save(output / name, arr)

        _save("embed.npy", "transformer.wte.weight", np.float16)
        _save("pos_embed.npy", "transformer.wpe.weight", np.float16)
        _save("ln_f_weight.npy", "transformer.ln_f.weight", np.float32)
        _save("ln_f_bias.npy", "transformer.ln_f.bias", np.float32)

        layers_dir = output / "layers"
        layers_dir.mkdir(exist_ok=True)

        for i in range(config["n_layers"]):
            layer_dir = layers_dir / str(i)
            layer_dir.mkdir(exist_ok=True)
            prefix = f"transformer.h.{i}."

            for store_name, (hf_suffix, dtype) in _WEIGHT_MAP.items():
                arr = sd[prefix + hf_suffix].float().numpy().astype(dtype)
                np.save(layer_dir / f"{store_name}.npy", arr)

        del model, sd

        (output / _SENTINEL).touch()

    def load_embed(self) -> np.ndarray:
        return np.load(self._dir / "embed.npy", mmap_mode="r")

    def load_pos_embed(self) -> np.ndarray:
        return np.load(self._dir / "pos_embed.npy", mmap_mode="r")

    def load_final_ln(self) -> tuple[np.ndarray, np.ndarray]:
        w = np.load(self._dir / "ln_f_weight.npy")
        b = np.load(self._dir / "ln_f_bias.npy")
        return w, b

    def load_layer(self, layer_idx: int) -> dict[str, np.ndarray]:
        """Load all weight tensors for one layer via memmap."""
        layer_dir = self._dir / "layers" / str(layer_idx)
        result = {}
        for fname in _LAYER_FILES:
            key = fname.replace(".npy", "")
            result[key] = np.load(layer_dir / fname, mmap_mode="r")
        return result

    def load_rows(self, layer_idx: int, key: str, indices: np.ndarray) -> np.ndarray:
        """Load specific rows of a weight matrix via memmap slicing."""
        path = self._dir / "layers" / str(layer_idx) / f"{key}.npy"
        full = np.load(path, mmap_mode="r")
        return np.array(full[indices])
