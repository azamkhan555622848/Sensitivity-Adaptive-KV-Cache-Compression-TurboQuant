"""Sensitivity-adaptive per-layer bit allocation."""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .compressors_v3 import MSECompressor

def calibrate(model, tokenizer, n_samples=16, max_seq_len=512, dataset_name="c4",
              seed: int = 42):
    """Capture KV tensors from calibration samples via forward hooks on
    ``k_proj`` and ``v_proj``. Hook-based capture bypasses any cache-related
    quirks (including Gemma-3's sliding attention which corrupts DynamicCache
    state on the final layers) and works for any architecture that exposes
    ``self_attn.k_proj`` / ``v_proj``.

    Returns {layer_idx: {"keys": Tensor, "values": Tensor}} where tensors are
    shaped (B, n_kv_heads, S, head_dim).
    """
    from datasets import load_dataset
    import random

    # Reproducible sample selection
    rng = random.Random(seed)

    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    elif dataset_name == "the_stack":
        ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Collect a pool, then sample seeded
    pool = []
    for item in ds:
        text = item.get("text", "") if isinstance(item, dict) else ""
        if isinstance(text, str) and len(text.strip()) > 100:
            pool.append(text)
        if len(pool) >= max(n_samples * 8, 64):
            break
    if len(pool) == 0:
        # Fallback: wikitext test split is a list of strings
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        pool = [t for t in ds["text"] if len(t.strip()) > 100]
    rng.shuffle(pool)
    texts = pool[:n_samples]

    encodings = tokenizer(texts, return_tensors="pt", truncation=True,
                          max_length=max_seq_len, padding=True)
    device = next(model.parameters()).device

    # Find the layers module. Supports standard HF layouts (model.model.layers)
    # and single-layer custom layouts.
    base = getattr(model, "model", model)
    layers_mod = getattr(base, "layers", None)
    if layers_mod is None:
        raise RuntimeError("Could not find layers attribute on model for calibration")

    # Infer KV head shape from the model config. For nested configs (Gemma-3
    # text_config), walk down.
    cfg = getattr(model.config, "text_config", model.config)
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)

    per_layer_k: Dict[int, List[torch.Tensor]] = {}
    per_layer_v: Dict[int, List[torch.Tensor]] = {}
    handles = []

    def make_hook(layer_idx: int, is_key: bool):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return output
            B, S, D_total = out.shape
            if D_total != n_kv_heads * head_dim:
                # Layer has a different head layout (e.g., a sparsified/pruned
                # attention block). Skip capture for this layer.
                return output
            x = out.view(B, S, n_kv_heads, head_dim).transpose(1, 2).contiguous()
            store = per_layer_k if is_key else per_layer_v
            store.setdefault(layer_idx, []).append(x.detach().to("cpu").float())
            return output
        return hook

    # Install hooks on every layer that has a standard k_proj/v_proj pair.
    captured_layer_ids = []
    for layer_idx, layer in enumerate(layers_mod):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if k_proj is None or v_proj is None:
            continue
        handles.append(k_proj.register_forward_hook(make_hook(layer_idx, True)))
        handles.append(v_proj.register_forward_hook(make_hook(layer_idx, False)))
        captured_layer_ids.append(layer_idx)

    try:
        with torch.no_grad():
            for i in range(0, len(texts), 4):
                batch_ids = encodings["input_ids"][i:i + 4].to(device)
                batch_mask = encodings["attention_mask"][i:i + 4].to(device)
                try:
                    model(batch_ids, attention_mask=batch_mask, use_cache=False)
                except Exception:
                    # Fall back to forward without attention_mask for models
                    # that reject batched padded input in this mode.
                    model(batch_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    result = {}
    for layer_idx in captured_layer_ids:
        ks = per_layer_k.get(layer_idx, [])
        vs = per_layer_v.get(layer_idx, [])
        if not ks or not vs:
            continue
        K = torch.cat(ks, dim=0)
        V = torch.cat(vs, dim=0)
        # Skip layers where capture produced NaN/Inf (e.g., pruned or no-op
        # attention layers). The adaptive allocator will simply ignore them.
        if not torch.isfinite(K).all() or not torch.isfinite(V).all():
            continue
        result[layer_idx] = {"keys": K, "values": V}
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
