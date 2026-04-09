import torch
from turboquant.compressors_v3 import TurboQuantV3

def test_layer_bits_override():
    comp = TurboQuantV3(head_dim=64, key_bits=4, value_bits=2, layer_idx=5, n_layers=32, protected_layers=0)
    assert comp.key_bits == 4
    assert comp.value_bits == 2
    comp2 = TurboQuantV3(head_dim=64, key_bits=4, value_bits=2, layer_idx=5, n_layers=32, protected_layers=0, layer_bits=(6, 3))
    assert comp2.key_bits == 6
    assert comp2.value_bits == 3

def test_layer_bits_compress_decompress():
    comp = TurboQuantV3(head_dim=64, key_bits=4, value_bits=2, layer_idx=0, n_layers=4, protected_layers=0, layer_bits=(6, 4))
    keys = torch.randn(1, 4, 32, 64)
    values = torch.randn(1, 4, 32, 64)
    ck, cv = comp.compress_kv(keys, values)
    dk, dv = comp.decompress_kv(ck, cv)
    assert dk.shape == keys.shape
    assert dv.shape == values.shape

from turboquant.adaptive import allocate_bits

def test_allocate_bits_respects_budget():
    sensitivity = {
        0: {2: 5.0, 3: 2.0, 4: 0.5, 6: 0.1, 8: 0.01},
        1: {2: 0.5, 3: 0.2, 4: 0.1, 6: 0.05, 8: 0.01},
        2: {2: 3.0, 3: 1.0, 4: 0.3, 6: 0.08, 8: 0.01},
        3: {2: 0.3, 3: 0.1, 4: 0.05, 6: 0.02, 8: 0.01},
    }
    allocation = allocate_bits(sensitivity, budget=4.0, bit_options=[2, 3, 4, 6, 8])
    assert len(allocation) == 4
    total_bits = sum(kb + vb for kb, vb in allocation.values())
    avg_bits = total_bits / (2 * len(allocation))
    assert avg_bits <= 4.01

def test_allocate_bits_sensitive_layers_get_more():
    sensitivity = {
        0: {2: 10.0, 4: 1.0, 8: 0.001},
        1: {2: 0.01, 4: 0.005, 8: 0.001},
    }
    allocation = allocate_bits(sensitivity, budget=5.0, bit_options=[2, 4, 8])
    kb0, vb0 = allocation[0]
    kb1, vb1 = allocation[1]
    assert (kb0 + vb0) >= (kb1 + vb1)

def test_allocate_bits_kv_split_format():
    """Test with separate key/value sensitivity — should give asymmetric allocation."""
    sensitivity = {
        0: {"key": {2: 10.0, 4: 1.0, 8: 0.001}, "value": {2: 0.1, 4: 0.01, 8: 0.001}},
        1: {"key": {2: 0.1, 4: 0.01, 8: 0.001}, "value": {2: 10.0, 4: 1.0, 8: 0.001}},
    }
    allocation = allocate_bits(sensitivity, budget=3.0, bit_options=[2, 4, 8])
    # Layer 0: keys are sensitive, values are not -> should get K=4, V=2
    kb0, vb0 = allocation[0]
    assert kb0 > vb0, f"Layer 0 should have higher key bits, got K={kb0}, V={vb0}"
    # Layer 1: opposite
    kb1, vb1 = allocation[1]
    assert vb1 > kb1, f"Layer 1 should have higher value bits, got K={kb1}, V={vb1}"
