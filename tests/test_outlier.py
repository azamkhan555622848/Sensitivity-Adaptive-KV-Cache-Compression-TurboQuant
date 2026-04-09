import torch
from turboquant.outlier import detect_outlier_channels, OutlierAwareMSECompressor

def test_detect_outlier_channels():
    kv_data = torch.randn(8, 4, 64, 128)
    kv_data[:, :, :, 0] *= 100.0
    kv_data[:, :, :, 1] *= 100.0
    mask = detect_outlier_channels(kv_data)
    assert mask.shape == (128,)
    assert mask[0] == True
    assert mask[1] == True
    assert mask.sum().item() < 128 * 0.1

def test_outlier_compressor_shape():
    outlier_mask = torch.zeros(128, dtype=torch.bool)
    outlier_mask[:4] = True
    comp = OutlierAwareMSECompressor(head_dim=128, bits=4, outlier_mask=outlier_mask, seed=42)
    states = torch.randn(1, 4, 32, 128)
    reconstructed = comp.decompress(comp.compress(states))
    assert reconstructed.shape == states.shape

def test_outlier_compressor_lower_error():
    states = torch.randn(1, 4, 32, 128)
    states[:, :, :, 0] *= 50.0
    states[:, :, :, 1] *= 50.0
    outlier_mask = torch.zeros(128, dtype=torch.bool)
    outlier_mask[:2] = True
    comp_outlier = OutlierAwareMSECompressor(head_dim=128, bits=4, outlier_mask=outlier_mask, seed=42)
    r_outlier = comp_outlier.decompress(comp_outlier.compress(states))
    mse_outlier = (states - r_outlier).pow(2).mean().item()
    from turboquant.compressors_v3 import MSECompressor
    comp_naive = MSECompressor(head_dim=128, bits=4, seed=42)
    r_naive = comp_naive.decompress(comp_naive.compress(states))
    mse_naive = (states - r_naive).pow(2).mean().item()
    assert mse_outlier < mse_naive
