"""PolarQuant: KV cache quantization using polar decomposition rotation."""
import torch
from ..lloyd_max import LloydMaxCodebook

def polar_rotation(weight: torch.Tensor) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    d = min(weight.shape)
    return (U[:, :d] @ Vh[:d, :]).to(weight.dtype)

class PolarQuantCompressor:
    def __init__(self, head_dim, bits, weight_matrix, seed=42, device="cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.Pi = polar_rotation(weight_matrix).to(device)
        if self.Pi.shape[0] != head_dim or self.Pi.shape[1] != head_dim:
            self.Pi = self.Pi[:head_dim, :head_dim]
        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states):
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()
        vec_norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)
        rotated = flat_norm @ self.Pi.T
        indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).to(torch.uint8)
        return {"indices": indices.reshape(B, H, S, D), "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S), "shape": (B, H, S, D)}

    @torch.no_grad()
    def decompress(self, compressed):
        B, H, S, D = compressed["shape"]
        N = B * H * S
        indices = compressed["indices"].reshape(N, D).long()
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()
        return (self.centroids[indices] @ self.Pi * vec_norms).reshape(B, H, S, D)
