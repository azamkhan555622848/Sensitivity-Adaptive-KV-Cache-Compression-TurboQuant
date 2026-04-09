import os
import torch
from turboquant.lloyd_max import LloydMaxCodebook

CACHE_DIR = os.path.expanduser("~/.cache/turboquant")

def test_codebook_caches_to_disk():
    cache_file = os.path.join(CACHE_DIR, "codebook_64_3.pt")
    if os.path.exists(cache_file):
        os.remove(cache_file)
    cb1 = LloydMaxCodebook(64, 3)
    assert os.path.exists(cache_file)
    cb2 = LloydMaxCodebook(64, 3)
    assert torch.allclose(cb1.centroids, cb2.centroids)
    assert torch.allclose(cb1.boundaries, cb2.boundaries)

def test_codebook_values_unchanged():
    cache_file = os.path.join(CACHE_DIR, "codebook_128_4.pt")
    if os.path.exists(cache_file):
        os.remove(cache_file)
    cb = LloydMaxCodebook(128, 4)
    assert cb.centroids.shape == (16,)
    assert cb.boundaries.shape == (15,)
    assert cb.distortion > 0
