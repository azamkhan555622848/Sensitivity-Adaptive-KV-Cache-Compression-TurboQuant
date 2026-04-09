"""Tests for CompressedCache with TurboQuantV3 as the compressor backend."""

import torch
from turboquant.cache import CompressedCache
from turboquant.compressors_v3 import TurboQuantV3


def make_tq_factory(key_bits=4, value_bits=4, residual_window=64, n_layers=4):
    def factory(layer_idx, head_dim, device):
        return TurboQuantV3(
            head_dim=head_dim, key_bits=key_bits, value_bits=value_bits,
            residual_window=0, layer_idx=layer_idx, n_layers=n_layers,
            protected_layers=0, seed=42, device=device,
        )
    return factory


def test_cache_basic_update():
    cache = CompressedCache(n_layers=4, head_dim=64, residual_window=32,
                            compressor_factory=make_tq_factory(n_layers=4))
    B, H, S, D = 1, 4, 16, 64
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)
    out_k, out_v = cache.update(keys, values, layer_idx=0)
    assert out_k.shape == (B, H, S, D)
    assert out_v.shape == (B, H, S, D)
    assert cache.get_seq_length(0) == S


def test_cache_incremental_growth():
    cache = CompressedCache(n_layers=4, head_dim=64, residual_window=32,
                            compressor_factory=make_tq_factory(n_layers=4))
    B, H, D = 1, 4, 64
    k1 = torch.randn(B, H, 16, D)
    v1 = torch.randn(B, H, 16, D)
    out_k, out_v = cache.update(k1, v1, layer_idx=0)
    assert out_k.shape[2] == 16

    k2 = torch.randn(B, H, 32, D)
    v2 = torch.randn(B, H, 32, D)
    out_k, out_v = cache.update(k2, v2, layer_idx=0)
    assert out_k.shape[2] == 48
    assert cache.get_seq_length(0) == 48


def test_cache_compression_info():
    cache = CompressedCache(n_layers=4, head_dim=64, residual_window=16,
                            compressor_factory=make_tq_factory(n_layers=4))
    B, H, D = 1, 4, 64
    keys = torch.randn(B, H, 64, D)
    values = torch.randn(B, H, 64, D)
    cache.update(keys, values, layer_idx=0)
    info = cache.get_compression_info()
    assert "compressed" in info
    assert "fp16" in info
