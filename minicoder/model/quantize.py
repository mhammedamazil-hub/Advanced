"""Symmetric per-row int8 quantization for the compact model.

2-D weights are quantized row-wise; 1-D vectors (norms, embeddings row-wise
too) stay in float32 (they are tiny and precision-sensitive).  This typically
shrinks weight storage ~4x with a measurable, small accuracy effect — the
honest numbers are produced by ``minicoder.evals.model_eval``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .transformer import MiniGPT, param_count


def _quantize_array(w: np.ndarray) -> dict[str, Any]:
    w32 = np.asarray(w, dtype=np.float32)
    max_abs = np.abs(w32).max(axis=1, keepdims=True)
    scale = np.maximum(max_abs / 127.0, 1e-12)
    q = np.clip(np.round(w32 / scale), -127, 127).astype(np.int8)
    return {"q": q, "scale": scale.astype(np.float32)}


def _dequantize_array(entry: dict[str, Any]) -> np.ndarray:
    return entry["q"].astype(np.float32) * entry["scale"]


def quantize_model(model: MiniGPT) -> dict[str, Any]:
    """Quantize every 2-D weight; 1-D vectors stay float32."""
    packed: dict[str, Any] = {}
    for name, w in model.params.items():
        if w.ndim == 2:
            packed[name] = _quantize_array(w)
        else:
            packed[name] = np.asarray(w, dtype=np.float32)
    return packed


def dequantize_model(packed: dict[str, Any], model: MiniGPT) -> MiniGPT:
    for name, entry in packed.items():
        if isinstance(entry, dict):
            model.params[name] = _dequantize_array(entry).astype(model.config.dtype)
        else:
            model.params[name] = entry.astype(model.config.dtype)
    return model


def save_quantized(model: MiniGPT, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    packed = quantize_model(model)
    flat: dict[str, np.ndarray] = {}
    for name, entry in packed.items():
        if isinstance(entry, dict):
            flat["q::" + name] = entry["q"]
            flat["s::" + name] = entry["scale"]
        else:
            flat["raw::" + name] = entry
    np.savez_compressed(path, **flat)


def size_report(model: MiniGPT) -> dict:
    params = param_count(model)
    fp32_bytes = params * 4
    quant_bytes = 0
    for _name, entry in quantize_model(model).items():
        if isinstance(entry, dict):
            quant_bytes += entry["q"].size + entry["scale"].size * 4
        else:
            quant_bytes += entry.size * 4
    return {
        "params": params,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": quant_bytes,
        "compression_ratio": round(fp32_bytes / max(1, quant_bytes), 2),
    }
