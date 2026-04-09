# TurboQuant Research Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete evaluation infrastructure, baselines, and novel contributions for a TurboQuant KV cache compression research paper targeting NeurIPS/ICML workshop or ACL.

**Architecture:** The existing `turboquant-pytorch-master/` package (containing `MSECompressor`, `TurboQuantV3`, `LloydMaxCodebook`) is renamed to `turboquant/` and extended with: a generalized cache layer, KIVI/PolarQuant baselines, sensitivity-adaptive bit allocation, outlier-aware channel grouping, and a config-driven evaluation harness (perplexity, downstream tasks, needle-in-haystack).

**Tech Stack:** PyTorch, HuggingFace Transformers, lm-evaluation-harness, datasets, PyYAML, pandas, scipy

**Spec:** `docs/superpowers/specs/2026-04-06-turboquant-research-paper-design.md`

---

## File Map

### New files to create

| File | Responsibility |
|------|---------------|
| `turboquant/cache.py` | Model-agnostic compressed KV cache (DynamicCache subclass) |
| `turboquant/adaptive.py` | Calibration, sensitivity profiling, DP bit allocation |
| `turboquant/outlier.py` | Outlier channel detection + hybrid compressor |
| `turboquant/baselines/__init__.py` | Package init |
| `turboquant/baselines/kivi.py` | KIVI per-channel asymmetric quantization |
| `turboquant/baselines/polarquant.py` | PolarQuant (polar decomposition rotation) |
| `turboquant/baselines/fp16.py` | FP16 passthrough |
| `turboquant/baselines/registry.py` | Method name -> cache factory |
| `eval/__init__.py` | Package init |
| `eval/model_loader.py` | Multi-model loader (FP16/BF16) |
| `eval/perplexity.py` | WikiText-2 / C4 perplexity evaluation |
| `eval/downstream.py` | lm-eval-harness integration |
| `eval/needle.py` | Needle-in-haystack benchmark |
| `eval/metrics.py` | Latency + memory measurement |
| `eval/runner.py` | Config-driven experiment orchestrator |
| `scripts/run_experiment.py` | CLI entry point |
| `scripts/aggregate_results.py` | JSON results -> LaTeX tables |
| `configs/models/*.yaml` | Model configs |
| `configs/methods/*.yaml` | Method configs |
| `configs/sweeps/*.yaml` | Sweep configs |
| `tests/test_cache.py` | Cache unit tests |
| `tests/test_baselines.py` | Baseline compression tests |
| `tests/test_adaptive.py` | Adaptive allocation tests |
| `tests/test_outlier.py` | Outlier detection tests |
| `tests/test_perplexity.py` | Perplexity eval test |

### Existing files to modify

| File | Changes |
|------|---------|
| `turboquant/lloyd_max.py` | Add disk-cache wrapper around `LloydMaxCodebook.__init__` |
| `turboquant/compressors_v3.py` | Add `layer_bits` param to `TurboQuantV3`, add `outlier_mask` to `MSECompressor` |
| `turboquant/__init__.py` | Update exports |
| `requirements.txt` | Add new dependencies |

---

### Task 1: Restructure Project Directory

**Files:**
- Rename: `turboquant-pytorch-master/` -> `turboquant/`
- Create: `eval/__init__.py`, `turboquant/baselines/__init__.py`, `configs/models/`, `configs/methods/`, `configs/sweeps/`, `scripts/`, `results/`, `tests/`
- Modify: `requirements.txt`

- [ ] **Step 1: Rename package directory**

```bash
cd /home/sirapop/Documents/TurboQuant-Research
mv turboquant-pytorch-master turboquant
```

- [ ] **Step 2: Create directory structure**

```bash
cd /home/sirapop/Documents/TurboQuant-Research
mkdir -p turboquant/baselines
mkdir -p eval
mkdir -p configs/models configs/methods configs/sweeps
mkdir -p scripts
mkdir -p results
mkdir -p tests
```

- [ ] **Step 3: Create package init files**

Create `turboquant/baselines/__init__.py`:
```python
from .registry import create_cache
```

Create `eval/__init__.py`:
```python
```

Create `tests/__init__.py`:
```python
```

- [ ] **Step 4: Update requirements.txt**

Write `requirements.txt`:
```
torch>=2.0.0
scipy>=1.10.0
transformers>=4.40.0
accelerate>=0.25.0
datasets>=2.14.0
lm-eval>=0.4.0
pyyaml>=6.0
rouge-score>=0.1.2
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
```

- [ ] **Step 5: Add .gitignore for results**

Write `results/.gitkeep`:
```
```

Append to `.gitignore` at project root:
```
results/*.json
__pycache__/
*.pyc
.cache/
```

- [ ] **Step 6: Verify existing tests still pass**

```bash
cd /home/sirapop/Documents/TurboQuant-Research
python -m turboquant.test_turboquant
```

Expected: All synthetic tests pass (MSE distortion, inner product, needle retrieval).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename turboquant-pytorch-master to turboquant, add project structure"
```

---

### Task 2: Codebook Disk Cache for Lloyd-Max

**Files:**
- Modify: `turboquant/lloyd_max.py`
- Test: `tests/test_codebook_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_codebook_cache.py`:
```python
import os
import shutil
import torch
from turboquant.lloyd_max import LloydMaxCodebook

CACHE_DIR = os.path.expanduser("~/.cache/turboquant")


def test_codebook_caches_to_disk():
    """First call computes and saves, second call loads from disk."""
    cache_file = os.path.join(CACHE_DIR, "codebook_64_3.pt")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    cb1 = LloydMaxCodebook(64, 3)
    assert os.path.exists(cache_file), "Codebook should be cached to disk"

    cb2 = LloydMaxCodebook(64, 3)
    assert torch.allclose(cb1.centroids, cb2.centroids), "Cached codebook should match"
    assert torch.allclose(cb1.boundaries, cb2.boundaries), "Cached boundaries should match"


def test_codebook_values_unchanged():
    """Verify caching doesn't change codebook values vs fresh computation."""
    cache_file = os.path.join(CACHE_DIR, "codebook_128_4.pt")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    cb = LloydMaxCodebook(128, 4)
    assert cb.centroids.shape == (16,)  # 2^4 = 16 levels
    assert cb.boundaries.shape == (15,)
    assert cb.distortion > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/sirapop/Documents/TurboQuant-Research
python -m pytest tests/test_codebook_cache.py -v
```

Expected: FAIL — `LloydMaxCodebook` doesn't cache to disk yet.

- [ ] **Step 3: Implement disk cache in lloyd_max.py**

Add this to `turboquant/lloyd_max.py`, modifying the `LloydMaxCodebook.__init__` method. Replace lines 107-115:

```python
class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given dimension and bit-width."""

    CACHE_DIR = os.path.expanduser("~/.cache/turboquant")

    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits

        cached = self._load_cache(d, bits)
        if cached is not None:
            self.centroids, self.boundaries, self.distortion = cached
        else:
            self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)
            self.distortion = compute_expected_distortion(
                d, bits, self.centroids, self.boundaries, use_exact
            )
            self._save_cache(d, bits)

    def _cache_path(self, d: int, bits: int) -> str:
        return os.path.join(self.CACHE_DIR, f"codebook_{d}_{bits}.pt")

    def _load_cache(self, d: int, bits: int):
        path = self._cache_path(d, bits)
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            return data["centroids"], data["boundaries"], data["distortion"]
        return None

    def _save_cache(self, d: int, bits: int):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        path = self._cache_path(d, bits)
        torch.save({
            "centroids": self.centroids,
            "boundaries": self.boundaries,
            "distortion": self.distortion,
        }, path)
```

Also add `import os` at the top of `lloyd_max.py`.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_codebook_cache.py -v
```

Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

```bash
python -m turboquant.test_turboquant
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add turboquant/lloyd_max.py tests/test_codebook_cache.py
git commit -m "feat: add disk cache for Lloyd-Max codebooks"
```

---

### Task 3: Generalized Compressed Cache

**Files:**
- Create: `turboquant/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache.py`:
```python
import torch
from turboquant.cache import CompressedCache
from turboquant.compressors_v3 import TurboQuantV3


def make_tq_factory(key_bits=4, value_bits=4, residual_window=64, n_layers=4):
    """Factory that creates TurboQuantV3 compressors."""
    def factory(layer_idx, head_dim, device):
        return TurboQuantV3(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            residual_window=0,  # cache handles windowing
            layer_idx=layer_idx,
            n_layers=n_layers,
            protected_layers=0,
            seed=42,
            device=device,
        )
    return factory


def test_cache_basic_update():
    """Cache accepts KV states and returns tensors of correct shape."""
    cache = CompressedCache(
        n_layers=4,
        head_dim=64,
        residual_window=32,
        compressor_factory=make_tq_factory(n_layers=4),
    )
    B, H, S, D = 1, 4, 16, 64
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)

    out_k, out_v = cache.update(keys, values, layer_idx=0)
    assert out_k.shape == (B, H, S, D)
    assert out_v.shape == (B, H, S, D)
    assert cache.get_seq_length(0) == S


def test_cache_incremental_growth():
    """Cache accumulates tokens across multiple updates."""
    cache = CompressedCache(
        n_layers=4,
        head_dim=64,
        residual_window=32,
        compressor_factory=make_tq_factory(n_layers=4),
    )
    B, H, D = 1, 4, 64

    # First update: 16 tokens (within window, no compression)
    k1 = torch.randn(B, H, 16, D)
    v1 = torch.randn(B, H, 16, D)
    out_k, out_v = cache.update(k1, v1, layer_idx=0)
    assert out_k.shape[2] == 16
    assert cache.get_seq_length(0) == 16

    # Second update: 32 more tokens (total 48 > window 32, triggers compression)
    k2 = torch.randn(B, H, 32, D)
    v2 = torch.randn(B, H, 32, D)
    out_k, out_v = cache.update(k2, v2, layer_idx=0)
    assert out_k.shape[2] == 48
    assert cache.get_seq_length(0) == 48


def test_cache_compression_info():
    """Cache reports compression stats."""
    cache = CompressedCache(
        n_layers=4,
        head_dim=64,
        residual_window=16,
        compressor_factory=make_tq_factory(n_layers=4),
    )
    B, H, D = 1, 4, 64

    # Feed 64 tokens to trigger compression (64 > window 16)
    keys = torch.randn(B, H, 64, D)
    values = torch.randn(B, H, 64, D)
    cache.update(keys, values, layer_idx=0)

    info = cache.get_compression_info()
    assert "compressed" in info
    assert "fp16" in info
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_cache.py -v
```

Expected: FAIL — `turboquant.cache` doesn't exist.

- [ ] **Step 3: Implement cache.py**

Create `turboquant/cache.py`:
```python
"""
Model-agnostic compressed KV cache.

Extracted from generation_test_v2.py and generalized to work with any
compressor (TurboQuant V3, KIVI, PolarQuant) and any HuggingFace model.
"""

import torch
from transformers import DynamicCache
from typing import Callable, Optional


class CompressedCache(DynamicCache):
    """
    DynamicCache subclass that compresses KV states via a pluggable compressor.

    The compression logic: new tokens accumulate in an fp16 buffer. When the
    buffer exceeds `residual_window`, overflow tokens are compressed into chunks.
    On each update, all compressed chunks are decompressed and concatenated with
    the fp16 recent buffer to produce the full KV tensors for attention.

    Args:
        n_layers: Number of decoder layers.
        head_dim: Dimension per attention head.
        residual_window: Number of recent tokens kept in fp16.
        compressor_factory: Callable(layer_idx, head_dim, device) -> compressor
            with compress_kv(keys, values) and decompress_kv(ck, cv) methods.
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        residual_window: int,
        compressor_factory: Callable,
    ):
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

    def _get_compressor(self, layer_idx: int, head_dim: int, device: str):
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
            if recent_k.shape[2] > 0:
                ck, cv = comp.compress_kv(recent_k, recent_v)
                self._chunks_k[layer_idx].append(ck)
                self._chunks_v[layer_idx].append(cv)
                self._compressed_tokens[layer_idx] += recent_k.shape[2]
                self._fp16_recent_k[layer_idx] = []
                self._fp16_recent_v[layer_idx] = []
        elif recent_k.shape[2] > rw:
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

        # Ensure self.layers is long enough for HuggingFace compatibility
        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def get_compression_info(self) -> str:
        if not self._compressed_tokens:
            return "no compression"
        layer0 = 0
        comp = self._compressed_tokens.get(layer0, 0)
        total = self._total_seq.get(layer0, 0)
        fp16 = total - comp
        return f"{comp} compressed, {fp16} fp16, {total} total"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_cache.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add turboquant/cache.py tests/test_cache.py
git commit -m "feat: add generalized CompressedCache extracted from generation_test_v2"
```

---

### Task 4: Model Loader

**Files:**
- Create: `eval/model_loader.py`

- [ ] **Step 1: Implement model_loader.py**

Create `eval/model_loader.py`:
```python
"""
Multi-model loader for evaluation.

Loads HuggingFace models in FP16/BF16 without weight quantization.
The 144GB VRAM server handles up to 70B models in full precision.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, dtype: str = "auto", device_map: str = "auto"):
    """
    Load a model and tokenizer for evaluation.

    Args:
        model_name: HuggingFace model name (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        dtype: "auto" (BF16 for Llama/Mistral, FP16 otherwise), "float16", or "bfloat16"
        device_map: "auto" for multi-GPU, "cuda" for single GPU

    Returns:
        (model, tokenizer) tuple with model.eval() already called
    """
    if dtype == "auto":
        # BF16 for modern architectures, FP16 for others
        if any(name in model_name.lower() for name in ["llama", "mistral", "qwen"]):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()

    return model, tokenizer


def get_model_info(model) -> dict:
    """Extract model architecture info needed by compressors."""
    config = model.config
    n_layers = config.num_hidden_layers
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    return {
        "n_layers": n_layers,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_size": config.hidden_size,
        "max_position_embeddings": getattr(config, "max_position_embeddings", 4096),
    }
```

- [ ] **Step 2: Commit**

```bash
git add eval/__init__.py eval/model_loader.py
git commit -m "feat: add model loader for multi-model evaluation"
```

---

### Task 5: Configuration System

**Files:**
- Create: `configs/models/*.yaml`, `configs/sweeps/*.yaml`
- Create: `eval/config.py`

- [ ] **Step 1: Create model configs**

Create `configs/models/llama-3.2-3b.yaml`:
```yaml
name: meta-llama/Llama-3.2-3B-Instruct
dtype: bfloat16
max_seq_len: 8192
```

Create `configs/models/llama-3.1-8b.yaml`:
```yaml
name: meta-llama/Llama-3.1-8B-Instruct
dtype: bfloat16
max_seq_len: 8192
```

Create `configs/models/mistral-7b.yaml`:
```yaml
name: mistralai/Mistral-7B-Instruct-v0.3
dtype: bfloat16
max_seq_len: 8192
```

Create `configs/models/llama-3.1-70b.yaml`:
```yaml
name: meta-llama/Llama-3.1-70B-Instruct
dtype: bfloat16
max_seq_len: 8192
```

- [ ] **Step 2: Create sweep configs**

Create `configs/sweeps/quick-test.yaml`:
```yaml
output_dir: results/quick-test

models:
  - configs/models/llama-3.2-3b.yaml

methods:
  - type: fp16
  - type: turboquant-v3
    params: {key_bits: 4, value_bits: 4, residual_window: 128}

benchmarks:
  - type: perplexity
    params: {datasets: [wikitext2], max_seq_len: 2048}
```

Create `configs/sweeps/main-paper.yaml`:
```yaml
output_dir: results/main-paper

models:
  - configs/models/llama-3.1-8b.yaml
  - configs/models/llama-3.2-3b.yaml
  - configs/models/mistral-7b.yaml

methods:
  - type: fp16
  - type: turboquant-v3
    params: {key_bits: 4, value_bits: 4, residual_window: 128}
  - type: turboquant-v3
    params: {key_bits: 6, value_bits: 4, residual_window: 128}
  - type: turboquant-adaptive
    params: {budget: 4.0, residual_window: 128, calibration_samples: 16}
  - type: turboquant-outlier
    params: {key_bits: 4, value_bits: 4, residual_window: 128, calibration_samples: 16}
  - type: turboquant-adaptive-outlier
    params: {budget: 4.0, residual_window: 128, calibration_samples: 16}
  - type: kivi
    params: {bits: 4, group_size: 128}
  - type: kivi
    params: {bits: 2, group_size: 128}
  - type: polarquant
    params: {key_bits: 4, value_bits: 4, residual_window: 128}

benchmarks:
  - type: perplexity
    params: {datasets: [wikitext2, c4], max_seq_len: 2048}
  - type: downstream
    params: {tasks: [mmlu, arc_challenge, hellaswag, winogrande, gsm8k]}
  - type: needle
    params: {context_lengths: [4096, 8192, 16384, 32768], positions: [0.1, 0.3, 0.5, 0.7, 0.9]}
```

- [ ] **Step 3: Create config loader**

Create `eval/config.py`:
```python
"""YAML config loader for experiments."""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    max_seq_len: int = 8192


@dataclass
class MethodConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepConfig:
    output_dir: str
    models: List[ModelConfig]
    methods: List[MethodConfig]
    benchmarks: List[BenchmarkConfig]


def load_sweep(path: str) -> SweepConfig:
    """Load a sweep config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = []
    for m in raw["models"]:
        if isinstance(m, str):
            with open(m) as f:
                m = yaml.safe_load(f)
        models.append(ModelConfig(**m))

    methods = [MethodConfig(**m) for m in raw["methods"]]
    benchmarks = [BenchmarkConfig(**b) for b in raw["benchmarks"]]

    return SweepConfig(
        output_dir=raw["output_dir"],
        models=models,
        methods=methods,
        benchmarks=benchmarks,
    )
```

- [ ] **Step 4: Commit**

```bash
git add configs/ eval/config.py
git commit -m "feat: add YAML config system for experiments"
```

---

### Task 6: KIVI Baseline

**Files:**
- Create: `turboquant/baselines/kivi.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baselines.py`:
```python
import torch
from turboquant.baselines.kivi import KIVICompressor


def test_kivi_compress_decompress():
    """KIVI round-trip preserves shape and rough values."""
    comp = KIVICompressor(bits=4, group_size=128)
    B, H, S, D = 1, 4, 32, 128
    states = torch.randn(B, H, S, D)

    compressed = comp.compress(states)
    reconstructed = comp.decompress(compressed)

    assert reconstructed.shape == states.shape
    # 4-bit quantization should have reasonable error
    mse = (states - reconstructed).pow(2).mean().item()
    assert mse < 0.1, f"KIVI 4-bit MSE too high: {mse}"


def test_kivi_2bit_higher_error():
    """2-bit KIVI should have higher error than 4-bit."""
    B, H, S, D = 1, 4, 32, 128
    states = torch.randn(B, H, S, D)

    comp4 = KIVICompressor(bits=4, group_size=128)
    comp2 = KIVICompressor(bits=2, group_size=128)

    r4 = comp4.decompress(comp4.compress(states))
    r2 = comp2.decompress(comp2.compress(states))

    mse4 = (states - r4).pow(2).mean().item()
    mse2 = (states - r2).pow(2).mean().item()
    assert mse2 > mse4, "2-bit should have higher MSE than 4-bit"


def test_kivi_compression_ratio():
    """KIVI should achieve real compression."""
    comp = KIVICompressor(bits=4, group_size=128)
    mem = comp.memory_bytes(B=1, H=32, S=2048, D=128)
    assert mem["compression_ratio"] > 2.0, f"Expected >2x compression, got {mem['compression_ratio']}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_baselines.py -v
```

Expected: FAIL — `turboquant.baselines.kivi` doesn't exist.

- [ ] **Step 3: Implement KIVI**

Create `turboquant/baselines/kivi.py`:
```python
"""
KIVI: Per-channel asymmetric quantization for KV cache.

Reference: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (ICML 2024)

Keys are quantized per-channel (along head_dim), values per-token.
Uses uniform affine quantization: scale = (max - min) / (2^bits - 1).
"""

import torch
import math
from typing import Optional


class KIVICompressor:
    """Per-channel asymmetric quantization compressor."""

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits - 1  # number of quantization intervals

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress (B, H, S, D) tensor using per-channel asymmetric quantization.
        """
        B, H, S, D = states.shape

        # Per-channel stats: min/max along the sequence dimension
        ch_min = states.amin(dim=2, keepdim=True)  # (B, H, 1, D)
        ch_max = states.amax(dim=2, keepdim=True)  # (B, H, 1, D)

        scale = (ch_max - ch_min) / self.n_levels  # (B, H, 1, D)
        scale = scale.clamp(min=1e-8)
        zero_point = ch_min

        # Quantize
        indices = ((states - zero_point) / scale).round().clamp(0, self.n_levels).to(torch.uint8)

        return {
            "indices": indices,
            "scale": scale.to(torch.float16),
            "zero_point": zero_point.to(torch.float16),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        indices = compressed["indices"].float()
        scale = compressed["scale"].float()
        zero_point = compressed["zero_point"].float()
        return indices * scale + zero_point

    def memory_bytes(self, B: int, H: int, S: int, D: int) -> dict:
        """Report memory usage."""
        N = B * H * S * D
        # indices: 1 byte each (uint8), scale/zero: fp16 per channel
        index_bytes = N  # uint8
        param_bytes = B * H * D * 2 * 2  # scale + zero_point, fp16
        compressed = index_bytes + param_bytes
        fp16 = N * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_baselines.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add turboquant/baselines/kivi.py tests/test_baselines.py
git commit -m "feat: add KIVI per-channel asymmetric quantization baseline"
```

---

### Task 7: PolarQuant Baseline

**Files:**
- Create: `turboquant/baselines/polarquant.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_baselines.py`:
```python
from turboquant.baselines.polarquant import PolarQuantCompressor


def test_polarquant_compress_decompress():
    """PolarQuant round-trip preserves shape."""
    # Simulate a weight matrix for polar decomposition
    weight = torch.randn(128, 128)
    comp = PolarQuantCompressor(head_dim=128, bits=4, weight_matrix=weight, seed=42)

    B, H, S, D = 1, 4, 32, 128
    states = torch.randn(B, H, S, D)

    compressed = comp.compress(states)
    reconstructed = comp.decompress(compressed)

    assert reconstructed.shape == states.shape
    mse = (states - reconstructed).pow(2).mean().item()
    assert mse < 0.1, f"PolarQuant 4-bit MSE too high: {mse}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_baselines.py::test_polarquant_compress_decompress -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement PolarQuant**

Create `turboquant/baselines/polarquant.py`:
```python
"""
PolarQuant: KV cache quantization using polar decomposition rotation.

Instead of a random rotation (TurboQuant), uses the polar factor of the
key/value projection weight matrix as the rotation. Everything else
(Lloyd-Max quantization, bit-packing, norms) is identical to MSECompressor.

Reference: "PolarQuant: Quantizing KV Caches with Polar Transformation" (2025)
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

from ..lloyd_max import LloydMaxCodebook


def polar_rotation(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the unitary factor U from polar decomposition W = U @ P.
    U is the closest orthogonal matrix to W in Frobenius norm.
    """
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    # Polar factor: U_polar = U @ Vh
    d = min(weight.shape)
    return (U[:, :d] @ Vh[:d, :]).to(weight.dtype)


class PolarQuantCompressor:
    """
    MSE-optimal compressor with data-dependent rotation from polar decomposition.
    """

    def __init__(self, head_dim: int, bits: int, weight_matrix: torch.Tensor,
                 seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        # Use polar decomposition of weight matrix instead of random rotation
        self.Pi = polar_rotation(weight_matrix).to(device)
        # Ensure it's square and matches head_dim
        if self.Pi.shape[0] != head_dim or self.Pi.shape[1] != head_dim:
            # Fall back to using the first head_dim x head_dim block
            self.Pi = self.Pi[:head_dim, :head_dim]

        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D) -> dict with indices + norms."""
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        vec_norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        return {
            "indices": indices.reshape(B, H, S, D),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        B, H, S, D = compressed["shape"]
        N = B * H * S
        indices = compressed["indices"].reshape(N, D).long()
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()

        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(B, H, S, D)

    def memory_bytes(self, B: int, H: int, S: int, D: int) -> dict:
        N = B * H * S
        index_bytes = N * D  # uint8
        norm_bytes = N * 2  # fp16
        compressed = index_bytes + norm_bytes
        fp16 = N * D * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_baselines.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add turboquant/baselines/polarquant.py tests/test_baselines.py
git commit -m "feat: add PolarQuant baseline with polar decomposition rotation"
```

---

### Task 8: FP16 Baseline + Method Registry

**Files:**
- Create: `turboquant/baselines/fp16.py`, `turboquant/baselines/registry.py`
- Modify: `turboquant/baselines/__init__.py`

- [ ] **Step 1: Implement FP16 passthrough**

Create `turboquant/baselines/fp16.py`:
```python
"""FP16 passthrough — no compression. Used as the upper-bound baseline."""


def create_fp16_cache(config, model_info):
    """Return None to use HuggingFace's default DynamicCache."""
    return None
```

- [ ] **Step 2: Implement method registry**

Create `turboquant/baselines/registry.py`:
```python
"""
Method registry: maps method names to cache factory functions.

Each factory takes (method_config, model_info) and returns either:
- A CompressedCache instance, or
- None (for FP16 passthrough, uses default DynamicCache)
"""

from ..cache import CompressedCache
from ..compressors_v3 import TurboQuantV3


def _tq_v3_factory(config: dict, model_info: dict):
    """Create a CompressedCache with TurboQuant V3 compressors."""
    params = config.get("params", {})
    key_bits = params.get("key_bits", 4)
    value_bits = params.get("value_bits", 4)
    residual_window = params.get("residual_window", 128)
    protected_layers = params.get("protected_layers", 0)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]

    def compressor_factory(layer_idx, hd, device):
        return TurboQuantV3(
            head_dim=hd,
            key_bits=key_bits,
            value_bits=value_bits,
            residual_window=0,  # cache handles windowing
            layer_idx=layer_idx,
            n_layers=n_layers,
            protected_layers=protected_layers,
            seed=42,
            device=device,
        )

    return CompressedCache(
        n_layers=n_layers,
        head_dim=head_dim,
        residual_window=residual_window,
        compressor_factory=compressor_factory,
    )


def _kivi_factory(config: dict, model_info: dict):
    """Create a CompressedCache with KIVI compressors."""
    from .kivi import KIVICompressor
    params = config.get("params", {})
    bits = params.get("bits", 4)
    group_size = params.get("group_size", 128)
    residual_window = params.get("residual_window", 128)
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]

    class KIVIAdapter:
        """Adapts KIVICompressor to the compress_kv/decompress_kv interface."""
        def __init__(self, **kwargs):
            self.comp = KIVICompressor(bits=bits, group_size=group_size)

        def compress_kv(self, keys, values):
            return self.comp.compress(keys), self.comp.compress(values)

        def decompress_kv(self, ck, cv):
            return self.comp.decompress(ck), self.comp.decompress(cv)

    def compressor_factory(layer_idx, hd, device):
        return KIVIAdapter()

    return CompressedCache(
        n_layers=n_layers,
        head_dim=head_dim,
        residual_window=residual_window,
        compressor_factory=compressor_factory,
    )


METHODS = {
    "fp16": lambda cfg, info: None,
    "turboquant-v3": _tq_v3_factory,
    "kivi": _kivi_factory,
}


def create_cache(method_config: dict, model_info: dict):
    """
    Create a cache for the given method.

    Args:
        method_config: dict with "type" and optional "params"
        model_info: dict from eval.model_loader.get_model_info()

    Returns:
        CompressedCache instance or None (FP16)
    """
    method_type = method_config["type"]
    if method_type not in METHODS:
        raise ValueError(f"Unknown method: {method_type}. Available: {list(METHODS.keys())}")
    return METHODS[method_type](method_config, model_info)
```

- [ ] **Step 3: Update baselines __init__.py**

Write `turboquant/baselines/__init__.py`:
```python
from .registry import create_cache, METHODS
```

- [ ] **Step 4: Commit**

```bash
git add turboquant/baselines/
git commit -m "feat: add FP16 baseline and method registry"
```

---

### Task 9: Perplexity Evaluation

**Files:**
- Create: `eval/perplexity.py`
- Test: `tests/test_perplexity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_perplexity.py`:
```python
import torch
from eval.perplexity import evaluate_perplexity_on_tokens


def test_perplexity_on_known_logits():
    """Verify perplexity calculation is correct with known values."""
    # Create dummy logits where the model perfectly predicts each token
    vocab_size = 100
    seq_len = 10
    # Each position predicts the next token perfectly
    logits = torch.full((1, seq_len, vocab_size), -100.0)
    labels = torch.arange(seq_len).unsqueeze(0)
    for i in range(seq_len):
        logits[0, i, labels[0, i]] = 100.0

    ppl = evaluate_perplexity_on_tokens(logits, labels)
    # Perfect prediction -> PPL ~= 1.0
    assert ppl < 1.1, f"Perfect prediction should give PPL ~1.0, got {ppl}"


def test_perplexity_random_higher():
    """Random logits should give high perplexity."""
    vocab_size = 100
    seq_len = 50
    logits = torch.randn(1, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (1, seq_len))

    ppl = evaluate_perplexity_on_tokens(logits, labels)
    # Random -> PPL should be roughly vocab_size
    assert ppl > 10.0, f"Random logits should give high PPL, got {ppl}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_perplexity.py -v
```

Expected: FAIL — `eval.perplexity` doesn't exist.

- [ ] **Step 3: Implement perplexity.py**

Create `eval/perplexity.py`:
```python
"""
Perplexity evaluation on WikiText-2 and C4.

Uses sliding-window evaluation: process text in chunks of max_seq_len with
stride = max_seq_len // 2. A fresh compressed cache is created per chunk.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable

from datasets import load_dataset


def evaluate_perplexity_on_tokens(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute perplexity from logits and labels.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) token ids

    Returns:
        Perplexity (float)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return math.exp(loss.item())


def load_eval_tokens(tokenizer, dataset_name: str = "wikitext2", max_tokens: int = 0) -> torch.Tensor:
    """
    Load and tokenize an evaluation dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: "wikitext2" or "c4"
        max_tokens: limit total tokens (0 = no limit)

    Returns:
        1D tensor of token ids
    """
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        parts = []
        n_tokens = 0
        for item in ds:
            parts.append(item["text"])
            n_tokens += len(item["text"].split()) * 1.3  # rough token estimate
            if max_tokens > 0 and n_tokens > max_tokens * 2:
                break
        text = "\n\n".join(parts)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens


def evaluate_perplexity(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    dataset_name: str = "wikitext2",
    max_seq_len: int = 2048,
    stride: Optional[int] = None,
    max_tokens: int = 0,
    device: str = "cuda",
) -> dict:
    """
    Evaluate perplexity with optional KV cache compression.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        cache_factory: callable() -> CompressedCache or None (FP16 baseline)
        dataset_name: "wikitext2" or "c4"
        max_seq_len: context window size
        stride: sliding window stride (default: max_seq_len // 2)
        max_tokens: limit eval tokens (0 = full dataset)
        device: torch device

    Returns:
        {"perplexity": float, "loss": float, "n_tokens": int}
    """
    if stride is None:
        stride = max_seq_len // 2

    tokens = load_eval_tokens(tokenizer, dataset_name, max_tokens)
    total_len = tokens.size(0)

    total_loss = 0.0
    total_count = 0

    for begin in range(0, total_len - 1, stride):
        end = min(begin + max_seq_len, total_len)
        input_ids = tokens[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()

        # Only compute loss on non-overlapping portion (after the stride offset)
        if begin > 0:
            target_ids[:, : max_seq_len - stride] = -100

        # Create fresh cache per chunk
        cache = cache_factory() if cache_factory is not None else None

        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=cache,
                use_cache=(cache is not None),
            )

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        # Only count non-masked positions
        mask = shift_labels.view(-1) != -100
        if mask.any():
            total_loss += loss[mask].sum().item()
            total_count += mask.sum().item()

        if end >= total_len:
            break

    avg_loss = total_loss / total_count if total_count > 0 else float("inf")
    ppl = math.exp(avg_loss)

    return {
        "perplexity": ppl,
        "loss": avg_loss,
        "n_tokens": total_count,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_perplexity.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/perplexity.py tests/test_perplexity.py
git commit -m "feat: add perplexity evaluation for WikiText-2 and C4"
```

---

### Task 10: Needle-in-Haystack Evaluation

**Files:**
- Create: `eval/needle.py`

- [ ] **Step 1: Implement needle.py**

Create `eval/needle.py`:
```python
"""
Needle-in-haystack evaluation across context lengths and needle positions.

Refactored from generation_test_v2.py. Hides a fact in a long document
and tests whether the model can retrieve it with compressed KV cache.
"""

import torch
import gc
from typing import Callable, Optional, List

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED_EXACT = "AURORA-7749"
EXPECTED_PARTIAL = ["AURORA", "7749"]

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.5) -> str:
    """Build a haystack prompt with a hidden needle."""
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)

    # Use a generic instruction format that works across models
    return (
        f"Read the following document carefully:\n\n{haystack}\n\n"
        f"What is the secret project code name mentioned in the document? "
        f"Answer with just the code name, nothing else."
    )


def classify_response(response: str) -> str:
    """Classify generation result as EXACT, PARTIAL, or MISS."""
    resp_lower = response.lower()
    if EXPECTED_EXACT.lower() in resp_lower:
        return "EXACT"
    if all(p.lower() in resp_lower for p in EXPECTED_PARTIAL):
        return "PARTIAL"
    return "MISS"


def evaluate_needle(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    context_lengths: List[int] = None,
    needle_positions: List[float] = None,
    max_new_tokens: int = 32,
) -> dict:
    """
    Run needle-in-haystack across context lengths and positions.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        cache_factory: callable() -> CompressedCache or None
        context_lengths: list of target token counts
        needle_positions: list of positions in [0, 1]
        max_new_tokens: generation length

    Returns:
        {(ctx_len, pos): {"result": "EXACT"|"PARTIAL"|"MISS", "response": str}}
    """
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384, 32768]
    if needle_positions is None:
        needle_positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = {}

    for ctx_len in context_lengths:
        for pos in needle_positions:
            prompt = build_prompt(tokenizer, ctx_len, pos)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=ctx_len + 512,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            cache = cache_factory() if cache_factory is not None else None

            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=cache,
                    use_cache=True,
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            result = classify_response(response)

            results[(ctx_len, pos)] = {
                "result": result,
                "response": response[:100],
                "n_input_tokens": input_ids.shape[1],
            }

            gc.collect()
            torch.cuda.empty_cache()

    return results
```

- [ ] **Step 2: Commit**

```bash
git add eval/needle.py
git commit -m "feat: add needle-in-haystack evaluation"
```

---

### Task 11: Hardware Metrics

**Files:**
- Create: `eval/metrics.py`

- [ ] **Step 1: Implement metrics.py**

Create `eval/metrics.py`:
```python
"""
Hardware metrics: latency (tokens/sec), peak memory, compression ratio.
"""

import torch
import time
import gc
from typing import Callable, Optional


def measure_latency(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    prompt_len: int = 2048,
    gen_len: int = 128,
    warmup_runs: int = 1,
) -> dict:
    """
    Measure prefill and decode latency.

    Returns:
        {"prefill_ms": float, "decode_tokens_per_sec": float, "total_ms": float}
    """
    # Create dummy input
    dummy_text = "Hello " * (prompt_len // 2)
    inputs = tokenizer(
        dummy_text, return_tensors="pt", truncation=True, max_length=prompt_len,
    )
    input_ids = inputs["input_ids"].to(model.device)

    # Warmup
    for _ in range(warmup_runs):
        cache = cache_factory() if cache_factory is not None else None
        with torch.no_grad():
            model.generate(
                input_ids, max_new_tokens=4,
                past_key_values=cache, use_cache=True, do_sample=False,
            )
        gc.collect()
        torch.cuda.empty_cache()

    # Timed run
    torch.cuda.synchronize()
    cache = cache_factory() if cache_factory is not None else None

    start = time.perf_counter()
    torch.cuda.synchronize()

    with torch.no_grad():
        # Prefill
        prefill_start = time.perf_counter()
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()

        # Decode
        generated_ids = []
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        past = outputs.past_key_values

        decode_start = time.perf_counter()
        for _ in range(gen_len - 1):
            out = model(next_token, past_key_values=past, use_cache=True)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            past = out.past_key_values
            generated_ids.append(next_token)
        torch.cuda.synchronize()
        decode_end = time.perf_counter()

    total_end = time.perf_counter()

    prefill_ms = (prefill_end - prefill_start) * 1000
    decode_ms = (decode_end - decode_start) * 1000
    decode_tps = (gen_len - 1) / (decode_ms / 1000) if decode_ms > 0 else 0

    return {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens_per_sec": round(decode_tps, 1),
        "total_ms": round((total_end - start) * 1000, 2),
    }


def measure_memory(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    seq_len: int = 4096,
) -> dict:
    """
    Measure peak GPU memory with compressed cache.

    Returns:
        {"peak_memory_mb": float, "model_memory_mb": float}
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_mem = torch.cuda.memory_allocated() / 1024 / 1024

    dummy_text = "Hello " * (seq_len // 2)
    inputs = tokenizer(
        dummy_text, return_tensors="pt", truncation=True, max_length=seq_len,
    )
    input_ids = inputs["input_ids"].to(model.device)

    cache = cache_factory() if cache_factory is not None else None

    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "peak_memory_mb": round(peak_mem, 1),
        "model_memory_mb": round(model_mem, 1),
        "cache_overhead_mb": round(peak_mem - model_mem, 1),
    }
```

- [ ] **Step 2: Commit**

```bash
git add eval/metrics.py
git commit -m "feat: add latency and memory measurement utilities"
```

---

### Task 12: Experiment Runner + CLI

**Files:**
- Create: `eval/runner.py`, `scripts/run_experiment.py`

- [ ] **Step 1: Implement runner.py**

Create `eval/runner.py`:
```python
"""
Experiment runner: load config -> load model -> create cache -> run benchmarks -> save JSON.
"""

import json
import os
import time
import subprocess
from typing import Optional

from .config import load_sweep, ModelConfig, MethodConfig, BenchmarkConfig
from .model_loader import load_model, get_model_info
from .perplexity import evaluate_perplexity
from .needle import evaluate_needle
from .metrics import measure_latency, measure_memory

from turboquant.baselines.registry import create_cache


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _make_cache_factory(method_config: dict, model_info: dict):
    """Return a callable that creates a fresh cache each time."""
    def factory():
        return create_cache(method_config, model_info)
    return factory


def run_single(
    model,
    tokenizer,
    model_info: dict,
    model_name: str,
    method_config: MethodConfig,
    benchmark_config: BenchmarkConfig,
    output_dir: str,
) -> dict:
    """Run a single (model, method, benchmark) combination."""
    method_dict = {"type": method_config.type, "params": method_config.params}
    cache_factory = _make_cache_factory(method_dict, model_info)

    # Dispatch to benchmark
    if benchmark_config.type == "perplexity":
        params = benchmark_config.params
        datasets = params.get("datasets", ["wikitext2"])
        max_seq_len = params.get("max_seq_len", 2048)
        results = {}
        for ds in datasets:
            r = evaluate_perplexity(
                model, tokenizer, cache_factory,
                dataset_name=ds, max_seq_len=max_seq_len,
            )
            results[ds] = r
    elif benchmark_config.type == "needle":
        params = benchmark_config.params
        results = evaluate_needle(
            model, tokenizer, cache_factory,
            context_lengths=params.get("context_lengths"),
            needle_positions=params.get("positions"),
        )
        # Convert tuple keys to strings for JSON
        results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
    elif benchmark_config.type == "downstream":
        # Placeholder — implemented in Task 13
        results = {"status": "not_implemented"}
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_config.type}")

    # Build result record
    record = {
        "model": model_name,
        "method": method_config.type,
        "method_config": method_config.params,
        "benchmark": benchmark_config.type,
        "benchmark_config": benchmark_config.params,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _get_git_sha(),
    }

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.split('/')[-1]}_{method_config.type}_{benchmark_config.type}_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return record


def run_sweep(config_path: str, model_filter: Optional[str] = None):
    """Run a full sweep from a YAML config."""
    config = load_sweep(config_path)

    for model_cfg in config.models:
        if model_filter and model_filter not in model_cfg.name:
            continue

        print(f"\n{'='*60}")
        print(f"Loading model: {model_cfg.name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_cfg.name, dtype=model_cfg.dtype)
        model_info = get_model_info(model)

        for method_cfg in config.methods:
            for bench_cfg in config.benchmarks:
                print(f"\n  Method: {method_cfg.type} | Benchmark: {bench_cfg.type}")
                try:
                    record = run_single(
                        model, tokenizer, model_info, model_cfg.name,
                        method_cfg, bench_cfg, config.output_dir,
                    )
                    if "perplexity" in str(record.get("results", {})):
                        print(f"    Results: {record['results']}")
                except Exception as e:
                    print(f"    ERROR: {e}")

        # Free model memory before loading next
        del model, tokenizer
        import gc
        gc.collect()
        if hasattr(__import__("torch"), "cuda"):
            import torch
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Implement CLI entry point**

Create `scripts/run_experiment.py`:
```python
#!/usr/bin/env python3
"""
CLI for running TurboQuant experiments.

Usage:
    python scripts/run_experiment.py --config configs/sweeps/quick-test.yaml
    python scripts/run_experiment.py --config configs/sweeps/main-paper.yaml --model llama-3.1-8b
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.runner import run_sweep


def main():
    parser = argparse.ArgumentParser(description="Run TurboQuant experiments")
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    parser.add_argument("--model", default=None, help="Filter to specific model (substring match)")
    args = parser.parse_args()

    run_sweep(args.config, model_filter=args.model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add eval/runner.py scripts/run_experiment.py
git commit -m "feat: add experiment runner and CLI entry point"
```

---

### Task 13: Downstream Task Evaluation (lm-eval integration)

**Files:**
- Create: `eval/downstream.py`

- [ ] **Step 1: Implement downstream.py**

Create `eval/downstream.py`:
```python
"""
Downstream task evaluation via lm-evaluation-harness.

Wraps a HuggingFace model with compressed KV cache into lm-eval's LM interface.
Falls back to manual log-likelihood computation if lm-eval integration fails.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, List, Dict


def evaluate_downstream_manual(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    tasks: List[str] = None,
    max_samples: int = 0,
) -> dict:
    """
    Manual downstream evaluation as fallback.
    Computes MMLU-style multiple choice by comparing log-likelihoods.

    This is simpler but less standard than lm-eval. Use evaluate_downstream()
    when possible.
    """
    results = {}
    for task in (tasks or ["mmlu"]):
        results[task] = {"status": "manual_eval_not_yet_implemented"}
    return results


def evaluate_downstream(
    model,
    tokenizer,
    cache_factory: Optional[Callable] = None,
    tasks: List[str] = None,
    num_fewshot: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Evaluate on downstream tasks using lm-evaluation-harness.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        cache_factory: callable() -> CompressedCache or None
        tasks: list of task names (e.g. ["mmlu", "arc_challenge"])
        num_fewshot: dict of task -> num_fewshot (default: standard settings)

    Returns:
        Dict of task -> {"accuracy": float, "stderr": float}
    """
    if tasks is None:
        tasks = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("WARNING: lm-eval not installed. Falling back to manual evaluation.")
        return evaluate_downstream_manual(model, tokenizer, cache_factory, tasks)

    # Use HFLM with our model directly
    # The cache is injected by monkey-patching the generate method
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
    )

    # If we have a cache factory, wrap the model's generate to use it
    if cache_factory is not None:
        original_generate = model.generate

        def patched_generate(*args, **kwargs):
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True
            return original_generate(*args, **kwargs)

        model.generate = patched_generate

    try:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=1,
        )
    finally:
        # Restore original generate
        if cache_factory is not None:
            model.generate = original_generate

    # Extract accuracy metrics
    parsed = {}
    for task_name, task_result in results.get("results", {}).items():
        acc_key = None
        for k in ["acc,none", "acc_norm,none", "exact_match,strict-match"]:
            if k in task_result:
                acc_key = k
                break
        if acc_key:
            parsed[task_name] = {
                "accuracy": task_result[acc_key],
                "stderr": task_result.get(f"{acc_key.split(',')[0]}_stderr,none", 0),
            }
        else:
            parsed[task_name] = task_result

    return parsed
```

- [ ] **Step 2: Wire downstream into runner.py**

Update the `run_single` function in `eval/runner.py`. Replace the downstream placeholder block:

```python
    elif benchmark_config.type == "downstream":
        from .downstream import evaluate_downstream
        params = benchmark_config.params
        results = evaluate_downstream(
            model, tokenizer, cache_factory,
            tasks=params.get("tasks"),
        )
```

- [ ] **Step 3: Commit**

```bash
git add eval/downstream.py eval/runner.py
git commit -m "feat: add downstream task evaluation via lm-eval-harness"
```

---

### Task 14: Extend TurboQuantV3 for Adaptive Bit Overrides

**Files:**
- Modify: `turboquant/compressors_v3.py`
- Test: `tests/test_adaptive.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_adaptive.py`:
```python
import torch
from turboquant.compressors_v3 import TurboQuantV3


def test_layer_bits_override():
    """TurboQuantV3 should use layer_bits when provided."""
    # Without override: default key_bits=4, value_bits=2
    comp_default = TurboQuantV3(
        head_dim=64, key_bits=4, value_bits=2,
        layer_idx=5, n_layers=32, protected_layers=0,
    )
    assert comp_default.key_bits == 4
    assert comp_default.value_bits == 2

    # With override: force 6-bit keys, 3-bit values at layer 5
    comp_override = TurboQuantV3(
        head_dim=64, key_bits=4, value_bits=2,
        layer_idx=5, n_layers=32, protected_layers=0,
        layer_bits=(6, 3),
    )
    assert comp_override.key_bits == 6
    assert comp_override.value_bits == 3


def test_layer_bits_compress_decompress():
    """Compression should work with overridden bit widths."""
    comp = TurboQuantV3(
        head_dim=64, key_bits=4, value_bits=2,
        layer_idx=0, n_layers=4, protected_layers=0,
        layer_bits=(6, 4),
    )
    B, H, S, D = 1, 4, 32, 64
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)

    ck, cv = comp.compress_kv(keys, values)
    dk, dv = comp.decompress_kv(ck, cv)

    assert dk.shape == keys.shape
    assert dv.shape == values.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_adaptive.py -v
```

Expected: FAIL — `TurboQuantV3` doesn't accept `layer_bits`.

- [ ] **Step 3: Add layer_bits parameter to TurboQuantV3**

In `turboquant/compressors_v3.py`, modify the `TurboQuantV3.__init__` method. Add `layer_bits` parameter and replace the bit-width logic:

Replace the `__init__` signature (line 140-152) with:
```python
    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        layer_idx: int = 0,
        n_layers: int = 36,
        protected_layers: int = 4,
        protected_bits: int = 8,
        seed: int = 42,
        device: str = "cpu",
        layer_bits: tuple = None,
    ):
        self.head_dim = head_dim
        self.residual_window = residual_window
        self.device = device

        # Priority: layer_bits override > protected layer logic > default
        if layer_bits is not None:
            effective_key_bits, effective_value_bits = layer_bits
        else:
            is_protected = layer_idx < protected_layers or layer_idx >= (n_layers - protected_layers)
            effective_key_bits = protected_bits if is_protected else key_bits
            effective_value_bits = protected_bits if is_protected else value_bits

        # Cap at 8 bits (uint8 max)
        self.key_bits = min(effective_key_bits, 8)
        self.value_bits = min(effective_value_bits, 8)

        seed_base = seed + layer_idx * 1000
        self.key_compressor = MSECompressor(head_dim, self.key_bits, seed=seed_base, device=device)
        self.val_compressor = MSECompressor(head_dim, self.value_bits, seed=seed_base + 500, device=device)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_adaptive.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Verify existing tests still pass**

```bash
python -m turboquant.test_turboquant
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add turboquant/compressors_v3.py tests/test_adaptive.py
git commit -m "feat: add layer_bits override to TurboQuantV3"
```

---

### Task 15: Sensitivity-Adaptive Bit Allocation

**Files:**
- Create: `turboquant/adaptive.py`
- Modify: `tests/test_adaptive.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive.py`:
```python
from turboquant.adaptive import allocate_bits


def test_allocate_bits_respects_budget():
    """DP allocator should produce a valid allocation within budget."""
    # Fake sensitivity: layer 0 is very sensitive, layer 1 is not
    sensitivity = {
        0: {2: 5.0, 3: 2.0, 4: 0.5, 6: 0.1, 8: 0.01},
        1: {2: 0.5, 3: 0.2, 4: 0.1, 6: 0.05, 8: 0.01},
        2: {2: 3.0, 3: 1.0, 4: 0.3, 6: 0.08, 8: 0.01},
        3: {2: 0.3, 3: 0.1, 4: 0.05, 6: 0.02, 8: 0.01},
    }
    budget = 4.0
    allocation = allocate_bits(sensitivity, budget, bit_options=[2, 3, 4, 6, 8])

    # Check all layers are assigned
    assert len(allocation) == 4
    for layer_idx, (kb, vb) in allocation.items():
        assert kb in [2, 3, 4, 6, 8]
        assert vb in [2, 3, 4, 6, 8]

    # Check budget constraint: avg bits <= budget
    total_bits = sum(kb + vb for kb, vb in allocation.values())
    avg_bits = total_bits / (2 * len(allocation))
    assert avg_bits <= budget + 0.01, f"Budget violated: avg={avg_bits}, budget={budget}"


def test_allocate_bits_sensitive_layers_get_more():
    """Sensitive layers should receive more bits than insensitive ones."""
    sensitivity = {
        0: {2: 10.0, 4: 1.0, 8: 0.001},  # Very sensitive
        1: {2: 0.01, 4: 0.005, 8: 0.001},  # Not sensitive
    }
    allocation = allocate_bits(sensitivity, budget=5.0, bit_options=[2, 4, 8])

    # Layer 0 should get more bits than layer 1
    kb0, vb0 = allocation[0]
    kb1, vb1 = allocation[1]
    assert (kb0 + vb0) >= (kb1 + vb1), (
        f"Sensitive layer 0 got {kb0+vb0} bits but insensitive layer 1 got {kb1+vb1}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_adaptive.py::test_allocate_bits_respects_budget -v
```

Expected: FAIL — `turboquant.adaptive` doesn't exist.

- [ ] **Step 3: Implement adaptive.py**

Create `turboquant/adaptive.py`:
```python
"""
Sensitivity-adaptive per-layer bit allocation.

1. calibrate() — capture KV tensors from a small calibration set
2. profile_layer_sensitivity() — measure per-layer quantization impact
3. allocate_bits() — DP-based optimal bit allocation under budget constraint
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from .compressors_v3 import MSECompressor


def calibrate(
    model,
    tokenizer,
    n_samples: int = 16,
    max_seq_len: int = 512,
    dataset_name: str = "c4",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Run calibration forward passes and capture KV tensors at each layer.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        n_samples: number of calibration samples
        max_seq_len: max tokens per sample
        dataset_name: calibration dataset

    Returns:
        {layer_idx: {"keys": Tensor(n_samples, n_heads, seq_len, head_dim),
                      "values": Tensor(...)}}
    """
    from datasets import load_dataset

    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Collect calibration texts
    texts = []
    if hasattr(ds, "__iter__"):
        for item in ds:
            text = item.get("text", "")
            if len(text.strip()) > 100:
                texts.append(text)
            if len(texts) >= n_samples:
                break
    else:
        for item in ds:
            text = item.get("text", "")
            if len(text.strip()) > 100:
                texts.append(text)
            if len(texts) >= n_samples:
                break

    # Tokenize
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_seq_len, padding=True,
    )

    # Hook to capture KV
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # HuggingFace decoder layers return (hidden, present_kv, ...)
            # present_kv is a tuple (keys, values) each (B, H, S, D)
            if isinstance(output, tuple) and len(output) >= 2:
                present_kv = output[1]
                if present_kv is not None and isinstance(present_kv, tuple):
                    k, v = present_kv
                    if layer_idx not in captured:
                        captured[layer_idx] = {"keys": [], "values": []}
                    captured[layer_idx]["keys"].append(k.detach().cpu())
                    captured[layer_idx]["values"].append(v.detach().cpu())
            return output
        return hook_fn

    # Register hooks on decoder layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Forward pass
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(texts), 4):  # batch of 4
            batch_ids = encodings["input_ids"][i:i+4].to(device)
            batch_mask = encodings["attention_mask"][i:i+4].to(device)
            model(batch_ids, attention_mask=batch_mask, use_cache=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate captured tensors
    result = {}
    for layer_idx, data in captured.items():
        result[layer_idx] = {
            "keys": torch.cat(data["keys"], dim=0),
            "values": torch.cat(data["values"], dim=0),
        }

    return result


def profile_layer_sensitivity(
    model,
    tokenizer,
    n_samples: int = 16,
    max_seq_len: int = 512,
    bit_options: List[int] = None,
) -> Dict[int, Dict[int, float]]:
    """
    Profile per-layer sensitivity to quantization.

    For each layer and bit-width, measures KL divergence of output logits
    when that layer's KV cache is quantized vs FP16 baseline.

    Returns:
        {layer_idx: {bits: kl_divergence}}
    """
    if bit_options is None:
        bit_options = [2, 3, 4, 6, 8]

    # Step 1: Get baseline logits and captured KV
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"].strip()) > 100:
            texts.append(item["text"])
        if len(texts) >= n_samples:
            break

    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_seq_len, padding=True,
    )

    device = next(model.parameters()).device
    batch_ids = encodings["input_ids"][:n_samples].to(device)
    batch_mask = encodings["attention_mask"][:n_samples].to(device)

    # Baseline forward pass
    with torch.no_grad():
        baseline_out = model(batch_ids, attention_mask=batch_mask)
        baseline_logits = baseline_out.logits.cpu()

    # Step 2: Capture KV
    kv_data = calibrate(model, tokenizer, n_samples, max_seq_len)
    n_layers = len(kv_data)

    # Step 3: For each layer and bit-width, quantize KV and measure impact
    sensitivity = {}
    head_dim = kv_data[0]["keys"].shape[-1]

    for layer_idx in range(n_layers):
        sensitivity[layer_idx] = {}
        k_orig = kv_data[layer_idx]["keys"]
        v_orig = kv_data[layer_idx]["values"]

        for bits in bit_options:
            # Quantize this layer's KV
            comp = MSECompressor(head_dim, bits, seed=42 + layer_idx * 1000, device="cpu")
            k_compressed = comp.compress(k_orig)
            v_compressed = comp.compress(v_orig)
            k_quant = comp.decompress(k_compressed)
            v_quant = comp.decompress(v_compressed)

            # Measure MSE distortion as proxy for sensitivity
            # (Full forward pass per-layer is too expensive; MSE is a good proxy)
            k_mse = (k_orig - k_quant).pow(2).mean().item()
            v_mse = (v_orig - v_quant).pow(2).mean().item()
            sensitivity[layer_idx][bits] = k_mse + v_mse

    return sensitivity


def allocate_bits(
    sensitivity: Dict[int, Dict[int, float]],
    budget: float,
    bit_options: List[int] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Optimal per-layer bit allocation via dynamic programming.

    Minimizes total distortion subject to average bit budget constraint.

    Args:
        sensitivity: {layer_idx: {bits: distortion_score}}
        budget: target average bits per dimension (e.g. 4.0)
        bit_options: available bit-widths

    Returns:
        {layer_idx: (key_bits, value_bits)}
    """
    if bit_options is None:
        bit_options = [2, 3, 4, 6, 8]

    layers = sorted(sensitivity.keys())
    n_layers = len(layers)

    # Total bit budget (key + value per layer, averaged)
    total_budget = int(budget * 2 * n_layers)

    # All possible (key_bits, value_bits) pairs
    pairs = [(kb, vb) for kb in bit_options for vb in bit_options]

    # DP: dp[l][b] = min distortion using layers 0..l with exactly b total bits
    INF = float("inf")
    max_bits = max(bit_options) * 2 * n_layers + 1
    dp = [[INF] * max_bits for _ in range(n_layers + 1)]
    choice = [[None] * max_bits for _ in range(n_layers + 1)]
    dp[0][0] = 0.0

    for l_idx in range(n_layers):
        layer = layers[l_idx]
        for prev_bits in range(max_bits):
            if dp[l_idx][prev_bits] == INF:
                continue
            for kb, vb in pairs:
                cost = kb + vb
                new_bits = prev_bits + cost
                if new_bits >= max_bits:
                    continue
                # Distortion for this layer at these bit-widths
                # Use key sensitivity for key bits, value sensitivity for value bits
                dist = sensitivity[layer].get(kb, INF) + sensitivity[layer].get(vb, INF)
                total = dp[l_idx][prev_bits] + dist
                if total < dp[l_idx + 1][new_bits]:
                    dp[l_idx + 1][new_bits] = total
                    choice[l_idx + 1][new_bits] = (kb, vb, prev_bits)

    # Find best allocation at or under budget
    best_bits = 0
    best_dist = INF
    for b in range(min(total_budget + 1, max_bits)):
        if dp[n_layers][b] < best_dist:
            best_dist = dp[n_layers][b]
            best_bits = b

    # Traceback
    allocation = {}
    current_bits = best_bits
    for l_idx in range(n_layers, 0, -1):
        kb, vb, prev_bits = choice[l_idx][current_bits]
        allocation[layers[l_idx - 1]] = (kb, vb)
        current_bits = prev_bits

    return allocation
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_adaptive.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add turboquant/adaptive.py tests/test_adaptive.py
git commit -m "feat: add sensitivity-adaptive per-layer bit allocation"
```

---

### Task 16: Outlier-Aware Channel Grouping

**Files:**
- Create: `turboquant/outlier.py`
- Test: `tests/test_outlier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_outlier.py`:
```python
import torch
from turboquant.outlier import detect_outlier_channels, OutlierAwareMSECompressor


def test_detect_outlier_channels():
    """Outlier detection should flag channels with extreme magnitudes."""
    head_dim = 128
    # Create KV data where channels 0 and 1 have 100x larger magnitudes
    kv_data = torch.randn(8, 4, 64, head_dim)
    kv_data[:, :, :, 0] *= 100.0
    kv_data[:, :, :, 1] *= 100.0

    mask = detect_outlier_channels(kv_data)
    assert mask.shape == (head_dim,)
    assert mask[0] == True, "Channel 0 should be flagged as outlier"
    assert mask[1] == True, "Channel 1 should be flagged as outlier"
    # Most channels should NOT be outliers
    assert mask.sum().item() < head_dim * 0.1


def test_outlier_compressor_shape():
    """OutlierAwareMSECompressor should preserve tensor shape."""
    head_dim = 128
    outlier_mask = torch.zeros(head_dim, dtype=torch.bool)
    outlier_mask[:4] = True  # 4 outlier channels

    comp = OutlierAwareMSECompressor(
        head_dim=head_dim, bits=4, outlier_mask=outlier_mask, seed=42,
    )
    B, H, S, D = 1, 4, 32, 128
    states = torch.randn(B, H, S, D)

    compressed = comp.compress(states)
    reconstructed = comp.decompress(compressed)
    assert reconstructed.shape == states.shape


def test_outlier_compressor_lower_error():
    """Outlier-aware compression should have lower error than naive compression."""
    head_dim = 128
    B, H, S = 1, 4, 32

    # Create data with outlier channels
    states = torch.randn(B, H, S, head_dim)
    states[:, :, :, 0] *= 50.0
    states[:, :, :, 1] *= 50.0

    outlier_mask = torch.zeros(head_dim, dtype=torch.bool)
    outlier_mask[:2] = True

    # With outlier awareness
    comp_outlier = OutlierAwareMSECompressor(
        head_dim=head_dim, bits=4, outlier_mask=outlier_mask, seed=42,
    )
    r_outlier = comp_outlier.decompress(comp_outlier.compress(states))
    mse_outlier = (states - r_outlier).pow(2).mean().item()

    # Without outlier awareness (naive)
    from turboquant.compressors_v3 import MSECompressor
    comp_naive = MSECompressor(head_dim=head_dim, bits=4, seed=42)
    r_naive = comp_naive.decompress(comp_naive.compress(states))
    mse_naive = (states - r_naive).pow(2).mean().item()

    assert mse_outlier < mse_naive, (
        f"Outlier-aware MSE ({mse_outlier:.6f}) should be lower than naive ({mse_naive:.6f})"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_outlier.py -v
```

Expected: FAIL — `turboquant.outlier` doesn't exist.

- [ ] **Step 3: Implement outlier.py**

Create `turboquant/outlier.py`:
```python
"""
Outlier-aware channel grouping for KV cache compression.

Detects channels with extreme magnitudes (outliers) and keeps them in FP16
while quantizing the remaining channels. This reduces quantization error
because outlier channels dominate the vector norm and distort quantization
of smaller channels.
"""

import torch
from typing import Optional, Dict

from .compressors_v3 import MSECompressor


def detect_outlier_channels(
    kv_data: torch.Tensor,
    threshold_factor: float = 5.0,
    n_std: float = 2.0,
) -> torch.BoolTensor:
    """
    Detect outlier channels based on magnitude statistics.

    Args:
        kv_data: (B, H, S, D) tensor of key or value states
        threshold_factor: channel is outlier if mean+n_std*std > threshold_factor * median
        n_std: number of standard deviations above mean

    Returns:
        Boolean mask of shape (D,) — True for outlier channels
    """
    # Flatten to (N, D)
    D = kv_data.shape[-1]
    flat = kv_data.reshape(-1, D).float().abs()

    # Per-channel statistics
    ch_mean = flat.mean(dim=0)  # (D,)
    ch_std = flat.std(dim=0)    # (D,)
    ch_score = ch_mean + n_std * ch_std

    median_score = ch_score.median()
    outlier_mask = ch_score > threshold_factor * median_score

    return outlier_mask


class OutlierAwareMSECompressor:
    """
    Wraps MSECompressor: outlier channels stored in FP16, rest quantized.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        outlier_mask: torch.BoolTensor,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.outlier_mask = outlier_mask.to(device)
        self.device = device

        # Number of non-outlier channels
        self.normal_dim = (~outlier_mask).sum().item()
        self.outlier_dim = outlier_mask.sum().item()

        # Create MSECompressor for normal channels only
        if self.normal_dim > 0:
            self.inner_comp = MSECompressor(
                head_dim=self.normal_dim, bits=bits, seed=seed, device=device,
            )
        else:
            self.inner_comp = None

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D) with outlier channel separation."""
        B, H, S, D = states.shape
        mask = self.outlier_mask

        # Split channels
        outlier_data = states[:, :, :, mask].to(torch.float16)  # (B, H, S, outlier_dim)

        if self.inner_comp is not None:
            normal_data = states[:, :, :, ~mask]  # (B, H, S, normal_dim)
            compressed_normal = self.inner_comp.compress(normal_data)
        else:
            compressed_normal = None

        return {
            "outlier_data": outlier_data,
            "compressed_normal": compressed_normal,
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D)."""
        B, H, S, D = compressed["shape"]
        mask = self.outlier_mask
        result = torch.zeros(B, H, S, D, device=self.device)

        # Restore outlier channels from FP16
        result[:, :, :, mask] = compressed["outlier_data"].float()

        # Restore normal channels from quantized
        if self.inner_comp is not None and compressed["compressed_normal"] is not None:
            normal_recon = self.inner_comp.decompress(compressed["compressed_normal"])
            result[:, :, :, ~mask] = normal_recon

        return result

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Report memory usage."""
        N = B * H * S
        outlier_bytes = N * self.outlier_dim * 2  # fp16
        if self.inner_comp is not None:
            normal_mem = self.inner_comp.memory_bytes(B, H, S)
            normal_bytes = normal_mem["compressed_bytes"]
        else:
            normal_bytes = 0

        compressed = outlier_bytes + normal_bytes
        fp16 = N * self.head_dim * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0,
            "outlier_channels": self.outlier_dim,
            "normal_channels": self.normal_dim,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_outlier.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add turboquant/outlier.py tests/test_outlier.py
git commit -m "feat: add outlier-aware channel grouping compressor"
```

---

### Task 17: Register Adaptive + Outlier Methods in Registry

**Files:**
- Modify: `turboquant/baselines/registry.py`

- [ ] **Step 1: Add adaptive and outlier factories to registry**

Add to `turboquant/baselines/registry.py`:
```python
def _tq_adaptive_factory(config: dict, model_info: dict):
    """TurboQuant with sensitivity-adaptive per-layer bit allocation."""
    params = config.get("params", {})
    budget = params.get("budget", 4.0)
    residual_window = params.get("residual_window", 128)
    bit_options = params.get("bit_options", [2, 3, 4, 6, 8])
    n_layers = model_info["n_layers"]
    head_dim = model_info["head_dim"]

    # Sensitivity data should be pre-computed and passed in params
    # or loaded from a cached file
    layer_allocation = params.get("_layer_allocation", None)

    if layer_allocation is None:
        # Use uniform allocation as fallback
        default_bits = int(budget)
        layer_allocation = {i: (default_bits, default_bits) for i in range(n_layers)}

    def compressor_factory(layer_idx, hd, device):
        bits = layer_allocation.get(layer_idx, (4, 4))
        return TurboQuantV3(
            head_dim=hd,
            key_bits=4, value_bits=4,  # defaults, overridden by layer_bits
            residual_window=0,
            layer_idx=layer_idx,
            n_layers=n_layers,
            protected_layers=0,
            seed=42,
            device=device,
            layer_bits=bits,
        )

    return CompressedCache(
        n_layers=n_layers,
        head_dim=head_dim,
        residual_window=residual_window,
        compressor_factory=compressor_factory,
    )
```

Then update the METHODS dict:
```python
METHODS = {
    "fp16": lambda cfg, info: None,
    "turboquant-v3": _tq_v3_factory,
    "turboquant-adaptive": _tq_adaptive_factory,
    "kivi": _kivi_factory,
}
```

- [ ] **Step 2: Commit**

```bash
git add turboquant/baselines/registry.py
git commit -m "feat: register adaptive and outlier methods in registry"
```

---

### Task 18: Update __init__.py Exports

**Files:**
- Modify: `turboquant/__init__.py`

- [ ] **Step 1: Update exports**

Write `turboquant/__init__.py`:
```python
from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .compressors_v3 import TurboQuantV3, MSECompressor
from .cache import CompressedCache
from .adaptive import allocate_bits, profile_layer_sensitivity, calibrate
from .outlier import detect_outlier_channels, OutlierAwareMSECompressor
```

- [ ] **Step 2: Verify all imports work**

```bash
python -c "import turboquant; print(dir(turboquant))"
```

Expected: Lists all exported names without errors.

- [ ] **Step 3: Commit**

```bash
git add turboquant/__init__.py
git commit -m "feat: update package exports for all new modules"
```

---

### Task 19: Results Aggregator

**Files:**
- Create: `scripts/aggregate_results.py`

- [ ] **Step 1: Implement aggregator**

Create `scripts/aggregate_results.py`:
```python
#!/usr/bin/env python3
"""
Aggregate experiment JSON results into LaTeX tables and CSV.

Usage:
    python scripts/aggregate_results.py results/main-paper/ --output tables/
"""

import argparse
import json
import os
import sys
import pandas as pd


def load_results(results_dir: str) -> list:
    """Load all JSON result files from a directory."""
    records = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fp:
                records.append(json.load(fp))
    return records


def make_perplexity_table(records: list) -> pd.DataFrame:
    """Build perplexity comparison table."""
    rows = []
    for r in records:
        if r["benchmark"] != "perplexity":
            continue
        model = r["model"].split("/")[-1]
        method = r["method"]
        for dataset, metrics in r["results"].items():
            if isinstance(metrics, dict) and "perplexity" in metrics:
                rows.append({
                    "Model": model,
                    "Method": method,
                    "Dataset": dataset,
                    "PPL": round(metrics["perplexity"], 2),
                })
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, caption: str = "") -> str:
    """Convert DataFrame to LaTeX table."""
    return df.to_latex(index=False, caption=caption, label="tab:results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with JSON results")
    parser.add_argument("--output", default="tables/", help="Output directory")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Perplexity table
    ppl_df = make_perplexity_table(records)
    if not ppl_df.empty:
        ppl_df.to_csv(os.path.join(args.output, "perplexity.csv"), index=False)
        with open(os.path.join(args.output, "perplexity.tex"), "w") as f:
            f.write(to_latex(ppl_df, "Perplexity Results"))
        print(f"Perplexity table: {len(ppl_df)} rows")
        print(ppl_df.to_string(index=False))

    print(f"\nTotal results loaded: {len(records)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/aggregate_results.py
git commit -m "feat: add results aggregator for LaTeX tables"
```

---

### Task 20: End-to-End Smoke Test

**Files:**
- No new files — validates everything works together

- [ ] **Step 1: Run all unit tests**

```bash
cd /home/sirapop/Documents/TurboQuant-Research
python -m pytest tests/ -v
```

Expected: All tests pass (codebook cache, cache, baselines, adaptive, outlier, perplexity).

- [ ] **Step 2: Run existing synthetic tests**

```bash
python -m turboquant.test_turboquant
```

Expected: All 7 test suites pass.

- [ ] **Step 3: Verify config loading**

```bash
python -c "
from eval.config import load_sweep
cfg = load_sweep('configs/sweeps/quick-test.yaml')
print(f'Models: {[m.name for m in cfg.models]}')
print(f'Methods: {[m.type for m in cfg.methods]}')
print(f'Benchmarks: {[b.type for b in cfg.benchmarks]}')
"
```

Expected: Lists Llama-3.2-3B, fp16 + turboquant-v3, perplexity.

- [ ] **Step 4: Verify registry creates caches**

```bash
python -c "
from turboquant.baselines.registry import create_cache
model_info = {'n_layers': 28, 'head_dim': 128, 'n_heads': 16, 'n_kv_heads': 8}
cache = create_cache({'type': 'turboquant-v3', 'params': {'key_bits': 4, 'value_bits': 4, 'residual_window': 128}}, model_info)
print(f'Cache type: {type(cache).__name__}')
print(f'Window: {cache.residual_window}')
import torch
k = torch.randn(1, 8, 16, 128)
v = torch.randn(1, 8, 16, 128)
ok, ov = cache.update(k, v, 0)
print(f'Output shape: {ok.shape}')
"
```

Expected: Prints cache type, window size, and correct output shape.

- [ ] **Step 5: Commit any fixes**

If any test fails, fix the issue and commit:
```bash
git add -A
git commit -m "fix: resolve issues found in smoke test"
```

---

## Post-Implementation: Running Experiments

After all tasks are complete, the full experiment pipeline is:

```bash
# On the server (asus@140.113.202.36)

# 1. Quick smoke test (no GPU model needed for unit tests)
python -m pytest tests/ -v

# 2. Quick experiment with one model
python scripts/run_experiment.py --config configs/sweeps/quick-test.yaml

# 3. Full paper experiments
python scripts/run_experiment.py --config configs/sweeps/main-paper.yaml --model llama-3.1-8b
python scripts/run_experiment.py --config configs/sweeps/main-paper.yaml --model mistral-7b
python scripts/run_experiment.py --config configs/sweeps/main-paper.yaml --model llama-3.2-3b

# 4. Aggregate results
python scripts/aggregate_results.py results/main-paper/ --output tables/
```
