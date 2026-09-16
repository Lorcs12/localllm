"""Feature extraction for autoregressive speculative decoding.

Distinct from the classification path's pooled_features (abi/ridge_reader.py):
there we pool over all tokens to get a document-level vector; here we extract
the last valid token's hidden state, which is what predicts the next token in
an autoregressive model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import OSDConfig


def last_token_features(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    *,
    layer_index: int = -1,
    use_layer_norm: bool = True,
) -> np.ndarray:
    """Extract the last valid token's hidden state from a single layer.

    Unlike pooled_features which averages over all tokens for classification,
    this extracts the causal representation at position t that predicts token
    t+1 — the signal the draft head needs for next-token prediction.
    """
    n_layers = len(hidden_states)
    if layer_index < 0:
        layer_index = n_layers + layer_index
    hs = hidden_states[layer_index]  # [batch, seq, D]

    if use_layer_norm:
        hs = F.layer_norm(hs.float(), (hs.shape[-1],))

    last_pos = attention_mask.sum(dim=1) - 1  # [batch]
    batch_idx = torch.arange(hs.shape[0], device=hs.device)
    features = hs[batch_idx, last_pos, :]  # [batch, D]
    return features.detach().cpu().numpy().astype(np.float64)


def extract_draft_features(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: OSDConfig,
) -> tuple[np.ndarray, Any]:
    """Run the draft model forward and return (features, model_output).

    The model_output is returned so the caller can reuse the KV cache
    for subsequent autoregressive steps without re-encoding the prefix.
    """
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )
    features = last_token_features(
        out.hidden_states,
        attention_mask,
        layer_index=config.layer_index,
        use_layer_norm=config.use_layer_norm,
    )
    return features, out


def get_lm_head_weights(model: Any) -> np.ndarray:
    """Extract the LM head weight matrix as [D, V] numpy float64.

    Most HuggingFace causal LMs store lm_head.weight as [V, D]; we transpose
    to match the ridge convention where W is [features, classes].
    """
    w = model.lm_head.weight.detach().cpu().float().numpy()  # [V, D]
    return w.T.astype(np.float64)  # [D, V]
