"""Method registry: maps method names to cache factory functions."""
from ..cache import CompressedCache
from ..compressors_v3 import TurboQuantV3

def _tq_v3_factory(config, model_info):
    params = config.get("params", {})
    key_bits = params.get("key_bits", 4)
    value_bits = params.get("value_bits", 4)
    residual_window = params.get("residual_window", 128)
    protected_layers = params.get("protected_layers", 0)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]
    def compressor_factory(layer_idx, hd, device):
        return TurboQuantV3(head_dim=hd, key_bits=key_bits, value_bits=value_bits,
                           residual_window=0, layer_idx=layer_idx, n_layers=n_layers,
                           protected_layers=protected_layers, seed=42, device=device)
    return CompressedCache(n_layers=n_layers, head_dim=head_dim,
                          residual_window=residual_window, compressor_factory=compressor_factory)

def _kivi_factory(config, model_info):
    from .kivi import KIVICompressor
    params = config.get("params", {})
    bits = params.get("bits", 4)
    group_size = params.get("group_size", 128)
    residual_window = params.get("residual_window", 128)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]
    class KIVIAdapter:
        def __init__(self, **kwargs):
            self.comp = KIVICompressor(bits=bits, group_size=group_size)
        def compress_kv(self, keys, values):
            return self.comp.compress(keys), self.comp.compress(values)
        def decompress_kv(self, ck, cv):
            return self.comp.decompress(ck), self.comp.decompress(cv)
    def compressor_factory(layer_idx, hd, device):
        return KIVIAdapter()
    return CompressedCache(n_layers=n_layers, head_dim=head_dim,
                          residual_window=residual_window, compressor_factory=compressor_factory)

def _tq_adaptive_factory(config, model_info):
    import json, os
    params = config.get("params", {})
    budget = params.get("budget", 4.0)
    residual_window = params.get("residual_window", 128)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]
    layer_allocation = params.get("_layer_allocation", None)
    # Load from calibration file if specified
    calibration_file = params.get("calibration_file", None)
    if layer_allocation is None and calibration_file and os.path.exists(calibration_file):
        with open(calibration_file) as f:
            cal_data = json.load(f)
        layer_allocation = {int(k): tuple(v) for k, v in cal_data["allocation"].items()}
    if layer_allocation is None:
        default_bits = int(budget)
        layer_allocation = {i: (default_bits, default_bits) for i in range(n_layers)}
    def compressor_factory(layer_idx, hd, device):
        bits = layer_allocation.get(layer_idx, (4, 4))
        return TurboQuantV3(head_dim=hd, key_bits=4, value_bits=4, residual_window=0,
                           layer_idx=layer_idx, n_layers=n_layers, protected_layers=0,
                           seed=42, device=device, layer_bits=bits)
    return CompressedCache(n_layers=n_layers, head_dim=head_dim,
                          residual_window=residual_window, compressor_factory=compressor_factory)

def _tq_outlier_factory(config, model_info):
    import json, os
    params = config.get("params", {})
    key_bits = params.get("key_bits", 4)
    value_bits = params.get("value_bits", 4)
    residual_window = params.get("residual_window", 128)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]
    # Load outlier profile
    outlier_file = params.get("outlier_file", None)
    layer_outliers = {}
    if outlier_file and os.path.exists(outlier_file):
        with open(outlier_file) as f:
            outlier_data = json.load(f)
        for k, v in outlier_data.get("layer_outliers", {}).items():
            layer_outliers[int(k)] = v
    # Also support adaptive allocation
    calibration_file = params.get("calibration_file", None)
    layer_allocation = None
    if calibration_file and os.path.exists(calibration_file):
        with open(calibration_file) as f:
            cal_data = json.load(f)
        layer_allocation = {int(k): tuple(v) for k, v in cal_data["allocation"].items()}
    import torch
    from ..outlier import OutlierAwareKVCompressor
    def compressor_factory(layer_idx, hd, device):
        if layer_allocation:
            kb, vb = layer_allocation.get(layer_idx, (key_bits, value_bits))
        else:
            kb, vb = key_bits, value_bits
        ol = layer_outliers.get(layer_idx, {})
        k_channels = ol.get("key_outlier_channels", [])
        v_channels = ol.get("value_outlier_channels", [])
        k_mask = torch.zeros(hd, dtype=torch.bool)
        v_mask = torch.zeros(hd, dtype=torch.bool)
        for ch in k_channels:
            k_mask[ch] = True
        for ch in v_channels:
            v_mask[ch] = True
        return OutlierAwareKVCompressor(
            head_dim=hd, key_bits=kb, value_bits=vb,
            key_outlier_mask=k_mask, value_outlier_mask=v_mask,
            seed=42 + layer_idx * 1000, device=device
        )
    return CompressedCache(n_layers=n_layers, head_dim=head_dim,
                          residual_window=residual_window, compressor_factory=compressor_factory)

METHODS = {
    "fp16": lambda cfg, info: None,
    "turboquant-v3": _tq_v3_factory,
    "turboquant-adaptive": _tq_adaptive_factory,
    "turboquant-outlier": _tq_outlier_factory,
    "turboquant-adaptive-outlier": _tq_outlier_factory,
    "kivi": _kivi_factory,
}

def create_cache(method_config, model_info):
    method_type = method_config["type"]
    if method_type not in METHODS:
        raise ValueError(f"Unknown method: {method_type}. Available: {list(METHODS.keys())}")
    return METHODS[method_type](method_config, model_info)
