"""Sensitivity-adaptive per-layer bit allocation."""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .compressors_v3 import MSECompressor

def calibrate(model, tokenizer, n_samples=16, max_seq_len=512, dataset_name="c4"):
    """Capture KV tensors from calibration samples via DynamicCache.
    Returns {layer_idx: {"keys": Tensor, "values": Tensor}}."""
    from datasets import load_dataset
    from transformers import DynamicCache
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) > 100:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
    device = next(model.parameters()).device
    all_keys = {}   # layer_idx -> list of tensors
    all_values = {}
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            batch_ids = encodings["input_ids"][i:i+4].to(device)
            batch_mask = encodings["attention_mask"][i:i+4].to(device)
            cache = DynamicCache()
            outputs = model(batch_ids, attention_mask=batch_mask, use_cache=True, past_key_values=cache)
            # Extract KV from cache — DynamicCache.layers[i] has .keys and .values
            for layer_idx in range(len(cache.layers)):
                layer_cache = cache.layers[layer_idx]
                k, v = layer_cache.keys, layer_cache.values
                if layer_idx not in all_keys:
                    all_keys[layer_idx] = []
                    all_values[layer_idx] = []
                all_keys[layer_idx].append(k.detach().cpu())
                all_values[layer_idx].append(v.detach().cpu())
    result = {}
    for layer_idx in all_keys:
        result[layer_idx] = {
            "keys": torch.cat(all_keys[layer_idx], dim=0),
            "values": torch.cat(all_values[layer_idx], dim=0)
        }
    return result

def profile_layer_sensitivity(model, tokenizer, n_samples=16, max_seq_len=512, bit_options=None):
    """Profile per-layer sensitivity with separate K/V curves.
    Returns {layer_idx: {"key": {bits: mse}, "value": {bits: mse}}}."""
    if bit_options is None:
        bit_options = [2, 3, 4, 6, 8]
    kv_data = calibrate(model, tokenizer, n_samples, max_seq_len)
    n_layers = len(kv_data)
    sensitivity = {}
    head_dim = kv_data[0]["keys"].shape[-1]
    for layer_idx in range(n_layers):
        sensitivity[layer_idx] = {"key": {}, "value": {}}
        k_orig = kv_data[layer_idx]["keys"]
        v_orig = kv_data[layer_idx]["values"]
        for bits in bit_options:
            comp = MSECompressor(head_dim, bits, seed=42 + layer_idx * 1000, device="cpu")
            k_compressed = comp.compress(k_orig)
            v_compressed = comp.compress(v_orig)
            k_quant = comp.decompress(k_compressed)
            v_quant = comp.decompress(v_compressed)
            sensitivity[layer_idx]["key"][bits] = (k_orig - k_quant).pow(2).mean().item()
            sensitivity[layer_idx]["value"][bits] = (v_orig - v_quant).pow(2).mean().item()
    return sensitivity

def allocate_bits(sensitivity, budget, bit_options=None):
    """DP-based optimal per-layer bit allocation with separate K/V sensitivity.
    Accepts either old format {layer: {bits: mse}} or new format {layer: {"key": {bits: mse}, "value": {bits: mse}}}.
    Returns {layer_idx: (key_bits, value_bits)}."""
    if bit_options is None:
        bit_options = [2, 3, 4, 6, 8]
    layers = sorted(sensitivity.keys())
    n_layers = len(layers)
    total_budget = int(budget * 2 * n_layers)
    pairs = [(kb, vb) for kb in bit_options for vb in bit_options]
    INF = float("inf")
    max_bits = max(bit_options) * 2 * n_layers + 1

    # Detect format: new has "key"/"value" sub-dicts
    sample = sensitivity[layers[0]]
    has_kv_split = isinstance(sample, dict) and "key" in sample and "value" in sample

    dp = [[INF] * max_bits for _ in range(n_layers + 1)]
    choice = [[None] * max_bits for _ in range(n_layers + 1)]
    dp[0][0] = 0.0
    for l_idx in range(n_layers):
        layer = layers[l_idx]
        if has_kv_split:
            k_sens = sensitivity[layer]["key"]
            v_sens = sensitivity[layer]["value"]
        else:
            k_sens = sensitivity[layer]
            v_sens = sensitivity[layer]
        for prev_bits in range(max_bits):
            if dp[l_idx][prev_bits] == INF:
                continue
            for kb, vb in pairs:
                new_bits = prev_bits + kb + vb
                if new_bits >= max_bits:
                    continue
                dist = k_sens.get(kb, INF) + v_sens.get(vb, INF)
                total = dp[l_idx][prev_bits] + dist
                if total < dp[l_idx + 1][new_bits]:
                    dp[l_idx + 1][new_bits] = total
                    choice[l_idx + 1][new_bits] = (kb, vb, prev_bits)
    best_bits, best_dist = 0, INF
    for b in range(min(total_budget + 1, max_bits)):
        if dp[n_layers][b] < best_dist:
            best_dist = dp[n_layers][b]
            best_bits = b
    allocation = {}
    current_bits = best_bits
    for l_idx in range(n_layers, 0, -1):
        kb, vb, prev_bits = choice[l_idx][current_bits]
        allocation[layers[l_idx - 1]] = (kb, vb)
        current_bits = prev_bits
    return allocation
