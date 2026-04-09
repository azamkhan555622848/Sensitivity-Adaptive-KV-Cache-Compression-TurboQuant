"""KIVI: Per-channel asymmetric quantization for KV cache (ICML 2024)."""
import torch
import math

class KIVICompressor:
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits - 1

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        B, H, S, D = states.shape
        ch_min = states.amin(dim=2, keepdim=True)
        ch_max = states.amax(dim=2, keepdim=True)
        scale = ((ch_max - ch_min) / self.n_levels).clamp(min=1e-8)
        zero_point = ch_min
        indices = ((states - zero_point) / scale).round().clamp(0, self.n_levels).to(torch.uint8)
        return {"indices": indices, "scale": scale.to(torch.float16), "zero_point": zero_point.to(torch.float16), "shape": (B, H, S, D)}

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        return compressed["indices"].float() * compressed["scale"].float() + compressed["zero_point"].float()

    def memory_bytes(self, B, H, S, D):
        N = B * H * S * D
        index_bytes = N
        param_bytes = B * H * D * 2 * 2
        compressed = index_bytes + param_bytes
        fp16 = N * 2
        return {"compressed_bytes": compressed, "fp16_bytes": fp16, "compression_ratio": fp16 / compressed if compressed > 0 else 0}
