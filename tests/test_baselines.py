import torch
from turboquant.baselines.kivi import KIVICompressor

def test_kivi_compress_decompress():
    comp = KIVICompressor(bits=4, group_size=128)
    states = torch.randn(1, 4, 32, 128)
    compressed = comp.compress(states)
    reconstructed = comp.decompress(compressed)
    assert reconstructed.shape == states.shape
    mse = (states - reconstructed).pow(2).mean().item()
    assert mse < 0.1

def test_kivi_2bit_higher_error():
    states = torch.randn(1, 4, 32, 128)
    r4 = KIVICompressor(bits=4).decompress(KIVICompressor(bits=4).compress(states))
    r2 = KIVICompressor(bits=2).decompress(KIVICompressor(bits=2).compress(states))
    assert (states - r2).pow(2).mean().item() > (states - r4).pow(2).mean().item()

def test_kivi_compression_ratio():
    comp = KIVICompressor(bits=4, group_size=128)
    mem = comp.memory_bytes(1, 32, 2048, 128)
    assert mem["compression_ratio"] > 1.5

from turboquant.baselines.polarquant import PolarQuantCompressor

def test_polarquant_compress_decompress():
    weight = torch.randn(128, 128)
    comp = PolarQuantCompressor(head_dim=128, bits=4, weight_matrix=weight, seed=42)
    states = torch.randn(1, 4, 32, 128)
    compressed = comp.compress(states)
    reconstructed = comp.decompress(compressed)
    assert reconstructed.shape == states.shape
    mse = (states - reconstructed).pow(2).mean().item()
    assert mse < 0.1
