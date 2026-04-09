"""
Generalized compressed KV cache.

Extracted from generation_test_v2.V3Cache — works with any compressor backend
that implements compress_kv(keys, values) and decompress_kv(ck, cv).
"""

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


class CompressedCache(DynamicCache):
    """
    A DynamicCache subclass that compresses KV states via a pluggable
    compressor backend.

    Args:
        n_layers: number of model layers
        head_dim: dimension per attention head
        residual_window: number of recent tokens kept in fp16 (0 = compress all)
        compressor_factory: callable(layer_idx, head_dim, device) -> compressor
            The returned compressor must have:
              - compress_kv(keys, values) -> (compressed_k, compressed_v)
              - decompress_kv(ck, cv) -> (keys, values)
    """

    def __init__(self, n_layers: int, head_dim: int, residual_window: int,
                 compressor_factory):
        super().__init__()
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.residual_window = residual_window
        self._compressor_factory = compressor_factory
        self._compressors = {}
        self._chunks_k = {}
        self._chunks_v = {}
        self._fp16_recent_k = {}
        self._fp16_recent_v = {}
        self._total_seq = {}
        self._compressed_tokens = {}

    def _get_compressor(self, layer_idx, head_dim, device):
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = self._compressor_factory(
                layer_idx, head_dim, str(device)
            )
        return self._compressors[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, S_new, D = key_states.shape
        device = key_states.device
        comp = self._get_compressor(layer_idx, D, device)

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
            self._compressed_tokens[layer_idx] = 0

        self._total_seq[layer_idx] += S_new

        # Add new tokens to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        # Concat recent buffer
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        # Compress tokens that exceed the residual window
        if rw == 0:
            # Compress all tokens (no fp16 window)
            if recent_k.shape[2] > 0:
                ck, cv = comp.compress_kv(recent_k, recent_v)
                self._chunks_k[layer_idx].append(ck)
                self._chunks_v[layer_idx].append(cv)
                self._compressed_tokens[layer_idx] += recent_k.shape[2]
                self._fp16_recent_k[layer_idx] = []
                self._fp16_recent_v[layer_idx] = []
        elif recent_k.shape[2] > rw:
            # Compress overflow, keep recent window in fp16
            overflow = recent_k.shape[2] - rw
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)
            self._compressed_tokens[layer_idx] += overflow

            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]

        # Decompress all chunks + concat with fp16 recent
        parts_k = []
        parts_v = []
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        # Add remaining fp16 recent tokens
        if self._fp16_recent_k[layer_idx]:
            recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
            recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
            parts_k.append(recent_k)
            parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2) if parts_k else key_states
        full_v = torch.cat(parts_v, dim=2) if parts_v else value_states

        # Grow self.layers for HuggingFace DynamicCache compatibility
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        return self._total_seq.get(layer_idx, 0)

    def get_compression_info(self):
        if not self._compressed_tokens:
            return "no compression"
        layer0 = 0
        comp = self._compressed_tokens.get(layer0, 0)
        total = self._total_seq.get(layer0, 0)
        fp16 = total - comp
        return f"{comp} compressed, {fp16} fp16, {total} total"
