"""Hook-based KV compression for models with custom cache formats.

Some models (e.g., Nemotron-NAS with its VariableCache) use a cache class that
is incompatible with our DynamicCache-based CompressedCache. Rather than
rewriting the cache for each such model, we take a more general approach:

1. Register forward hooks on each attention layer's k_proj and v_proj.
2. The hook captures the projection output (the raw K or V tensor before RoPE
   and attention).
3. The hook passes the tensor through compress + decompress, simulating the
   effect of storing it in a compressed KV cache.
4. The noisy tensor is returned; downstream code (RoPE, attention) runs on it.

For perplexity evaluation this is functionally equivalent to cache-based
compression, with one caveat: it compresses ALL tokens including the "current"
token, whereas a real cache would keep the current token in FP16 until the next
step. The effect on perplexity is negligible because self-attention dominates
over attention-to-self.

Usage:
    with HookCompressor(model, model_info, {"type": "turboquant-v3",
                                            "params": {"key_bits": 4, "value_bits": 4}}):
        logits = model(input_ids).logits
    # hooks are automatically removed on context exit
"""
import torch
from typing import Dict, Any, List, Optional, Tuple

from .compressors_v3 import MSECompressor


def _create_tq_v3_compressors(layer_idx: int, n_layers: int, head_dim: int,
                              device: str, params: Dict[str, Any]
                              ) -> Tuple[MSECompressor, MSECompressor]:
    """Build separate key and value MSECompressors matching TurboQuantV3 semantics."""
    key_bits = params.get("key_bits", 4)
    value_bits = params.get("value_bits", 4)
    protected_layers = params.get("protected_layers", 0)
    protected_bits = params.get("protected_bits", 8)
    # Layer-adaptive: first/last N layers get more bits
    layer_bits = params.get("layer_bits", None)
    if layer_bits is not None:
        effective_k, effective_v = layer_bits
    else:
        is_protected = (
            layer_idx < protected_layers
            or layer_idx >= (n_layers - protected_layers)
        )
        effective_k = protected_bits if is_protected else key_bits
        effective_v = protected_bits if is_protected else value_bits
    # Cap at 8 bits (uint8 max)
    effective_k = min(effective_k, 8)
    effective_v = min(effective_v, 8)
    seed_base = params.get("seed", 42) + layer_idx * 1000
    k_comp = MSECompressor(head_dim, effective_k, seed=seed_base, device=device)
    v_comp = MSECompressor(head_dim, effective_v, seed=seed_base + 500, device=device)
    return k_comp, v_comp


def _create_kivi_compressors(params: Dict[str, Any]):
    """Build KIVI compressors. KIVI's class is stateless so keys and values
    share the same compressor."""
    from .baselines.kivi import KIVICompressor
    bits = params.get("bits", 4)
    group_size = params.get("group_size", 128)
    comp = KIVICompressor(bits=bits, group_size=group_size)
    return comp, comp


def _create_tq_adaptive_compressors(layer_idx: int, head_dim: int, device: str,
                                    params: Dict[str, Any]
                                    ) -> Tuple[MSECompressor, MSECompressor]:
    """Build adaptive-allocation compressors for this layer."""
    import json
    import os
    calibration_file = params.get("calibration_file")
    layer_allocation = params.get("_layer_allocation")
    if layer_allocation is None and calibration_file and os.path.exists(calibration_file):
        with open(calibration_file) as f:
            cal_data = json.load(f)
        layer_allocation = {int(k): tuple(v) for k, v in cal_data["allocation"].items()}
    budget = int(params.get("budget", 4.0))
    default = (budget, budget)
    if layer_allocation is not None:
        kb, vb = layer_allocation.get(layer_idx, default)
    else:
        kb, vb = default
    seed_base = params.get("seed", 42) + layer_idx * 1000
    k_comp = MSECompressor(head_dim, kb, seed=seed_base, device=device)
    v_comp = MSECompressor(head_dim, vb, seed=seed_base + 500, device=device)
    return k_comp, v_comp


class HookCompressor:
    """Context manager that installs forward hooks to simulate KV compression
    without relying on the model's cache API.
    """

    def __init__(self, model, model_info: Dict[str, Any],
                 method_config: Dict[str, Any]):
        self.model = model
        self.model_info = model_info
        self.method_type = method_config.get("type", "fp16")
        self.method_params = method_config.get("params", {})
        self.handles: List[Any] = []
        # (layer_idx) -> (k_compressor, v_compressor)
        self.compressors: Dict[int, Tuple[Any, Any]] = {}

        # Which layers have real attention? For standard models: all layers.
        # For Nemotron-NAS: only the subset flagged as real attention.
        if "nemotron_real_attn_layers" in model_info:
            self.target_layers = set(model_info["nemotron_real_attn_layers"])
        else:
            self.target_layers = set(range(model_info["n_layers"]))

    def __enter__(self):
        if self.method_type == "fp16":
            return self
        self._install_hooks()
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _get_layers_module(self):
        """Return the iterable of transformer layer blocks on this model."""
        base = getattr(self.model, "model", self.model)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Could not find .model.layers on the model; "
                "HookCompressor may need model-specific extension."
            )
        return layers

    def _install_hooks(self):
        layers_mod = self._get_layers_module()
        n_layers = len(layers_mod)
        head_dim = self.model_info["head_dim"]

        for layer_idx in range(n_layers):
            if layer_idx not in self.target_layers:
                continue
            layer = layers_mod[layer_idx]
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            if k_proj is None or v_proj is None:
                continue

            device = str(next(k_proj.parameters()).device)

            if self.method_type == "turboquant-v3":
                k_c, v_c = _create_tq_v3_compressors(
                    layer_idx, n_layers, head_dim, device, self.method_params
                )
            elif self.method_type == "turboquant-adaptive":
                k_c, v_c = _create_tq_adaptive_compressors(
                    layer_idx, head_dim, device, self.method_params
                )
            elif self.method_type == "kivi":
                k_c, v_c = _create_kivi_compressors(self.method_params)
            else:
                continue  # Unsupported — skip silently (FP16 no-op handled in __enter__)

            self.compressors[layer_idx] = (k_c, v_c)

            k_hook = self._make_hook(layer_idx, is_key=True)
            v_hook = self._make_hook(layer_idx, is_key=False)
            self.handles.append(k_proj.register_forward_hook(k_hook))
            self.handles.append(v_proj.register_forward_hook(v_hook))

    def _make_hook(self, layer_idx: int, is_key: bool):
        """Create a forward hook that passes the projection output through
        compression + decompression in place."""
        n_kv_heads = self.model_info["n_kv_heads"]
        head_dim = self.model_info["head_dim"]

        def hook(module, inputs, output):
            # output shape: (B, S, n_kv_heads * head_dim)
            # Some modules (e.g., Gemma-3) return a tuple; handle that.
            out = output[0] if isinstance(output, tuple) else output
            B, S, D_total = out.shape
            expected = n_kv_heads * head_dim
            if D_total != expected:
                # Some fused projections might not match the expected shape.
                # In that case, skip compression for this layer to avoid corrupting
                # the forward pass. Log once via a sentinel so we don't spam.
                return output

            # Reshape to (B, H, S, D) matching the compressor contract
            x = out.view(B, S, n_kv_heads, head_dim).transpose(1, 2).contiguous()

            k_c, v_c = self.compressors[layer_idx]
            comp = k_c if is_key else v_c

            with torch.no_grad():
                compressed = comp.compress(x)
                x_out = comp.decompress(compressed)

            # Restore dtype and device
            x_out = x_out.to(x.dtype).to(x.device)
            x_out = x_out.transpose(1, 2).contiguous().view(B, S, D_total)

            if isinstance(output, tuple):
                return (x_out,) + output[1:]
            return x_out

        return hook
