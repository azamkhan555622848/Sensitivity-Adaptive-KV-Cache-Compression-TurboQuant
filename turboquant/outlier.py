"""Outlier-aware channel grouping for KV cache compression."""
import torch
from typing import Optional, Dict
from .compressors_v3 import MSECompressor

def detect_outlier_channels(kv_data, threshold_factor=5.0, n_std=2.0):
    D = kv_data.shape[-1]
    flat = kv_data.reshape(-1, D).float().abs()
    ch_mean = flat.mean(dim=0)
    ch_std = flat.std(dim=0)
    ch_score = ch_mean + n_std * ch_std
    median_score = ch_score.median()
    return ch_score > threshold_factor * median_score

class OutlierAwareMSECompressor:
    def __init__(self, head_dim, bits, outlier_mask, seed=42, device="cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.outlier_mask = outlier_mask.to(device)
        self.device = device
        self.normal_dim = (~outlier_mask).sum().item()
        self.outlier_dim = outlier_mask.sum().item()
        if self.normal_dim > 0:
            self.inner_comp = MSECompressor(head_dim=self.normal_dim, bits=bits, seed=seed, device=device)
        else:
            self.inner_comp = None

    @torch.no_grad()
    def compress(self, states):
        B, H, S, D = states.shape
        mask = self.outlier_mask
        outlier_data = states[:, :, :, mask].to(torch.float16)
        compressed_normal = self.inner_comp.compress(states[:, :, :, ~mask]) if self.inner_comp else None
        return {"outlier_data": outlier_data, "compressed_normal": compressed_normal, "shape": (B, H, S, D)}

    @torch.no_grad()
    def decompress(self, compressed):
        B, H, S, D = compressed["shape"]
        result = torch.zeros(B, H, S, D, device=self.device)
        result[:, :, :, self.outlier_mask] = compressed["outlier_data"].float()
        if self.inner_comp and compressed["compressed_normal"] is not None:
            result[:, :, :, ~self.outlier_mask] = self.inner_comp.decompress(compressed["compressed_normal"])
        return result

    def memory_bytes(self, B, H, S):
        N = B * H * S
        outlier_bytes = N * self.outlier_dim * 2
        normal_bytes = self.inner_comp.memory_bytes(B, H, S)["compressed_bytes"] if self.inner_comp else 0
        compressed = outlier_bytes + normal_bytes
        fp16 = N * self.head_dim * 2
        return {"compressed_bytes": compressed, "fp16_bytes": fp16,
                "compression_ratio": fp16 / compressed if compressed > 0 else 0,
                "outlier_channels": self.outlier_dim, "normal_channels": self.normal_dim}


class OutlierAwareKVCompressor:
    """Wraps OutlierAwareMSECompressor to provide compress_kv/decompress_kv interface
    compatible with CompressedCache."""

    def __init__(self, head_dim, key_bits, value_bits, key_outlier_mask, value_outlier_mask,
                 seed=42, device="cpu"):
        self.key_comp = OutlierAwareMSECompressor(
            head_dim, key_bits, key_outlier_mask, seed=seed, device=device
        )
        self.val_comp = OutlierAwareMSECompressor(
            head_dim, value_bits, value_outlier_mask, seed=seed + 500, device=device
        )

    def compress_kv(self, keys, values):
        return self.key_comp.compress(keys), self.val_comp.compress(values)

    def decompress_kv(self, ck, cv):
        return self.key_comp.decompress(ck), self.val_comp.decompress(cv)
