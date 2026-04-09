# TurboQuant Research Paper — Design Specification

## 1. Problem Statement

TurboQuant (ICLR 2026) introduced vector quantization for LLM KV cache compression using random rotation + Lloyd-Max quantization. The paper's second stage (QJL residual correction) was shown by 6+ independent teams to hurt KV cache quality because softmax exponentially amplifies QJL's variance.

Community implementations (V3) fixed this by dropping QJL, adding asymmetric K/V bit allocation, residual windowing, and layer-adaptive precision. However, no rigorous academic evaluation exists: the only tests are needle-in-haystack on a single model (Qwen2.5-3B).

**This paper provides:**
1. The first comprehensive cross-scale benchmark of TurboQuant V3 (3B-70B models, standard benchmarks, multiple baselines)
2. A novel sensitivity-adaptive per-layer bit allocation algorithm
3. A complementary outlier-aware channel grouping technique

**Target venue:** NeurIPS/ICML workshop or ACL

## 2. Existing Codebase

Location: `/home/sirapop/Documents/TurboQuant-Research/turboquant-pytorch-master/`

### Core modules (no changes needed)

| File | Key Classes | Role |
|------|-------------|------|
| `turboquant.py` | `TurboQuantMSE`, `TurboQuantProd`, `TurboQuantKVCache` | V1/V2 reference implementations |
| `compressors.py` | `TurboQuantCompressorV2`, `TurboQuantCompressorMSE` | V2 compressors (MSE+QJL for keys, MSE-only for values) |
| `lloyd_max.py` | `LloydMaxCodebook`, `solve_lloyd_max` | Lloyd-Max optimal scalar quantizer solver |
| `test_turboquant.py` | — | 7 synthetic algorithm test suites |

### Modules to extend

| File | Key Classes | Changes Needed |
|------|-------------|----------------|
| `compressors_v3.py` | `MSECompressor`, `TurboQuantV3` | Add `layer_bits` parameter override, add `outlier_mask` support to `MSECompressor.compress()` |
| `__init__.py` | — | Update exports for new modules |

### Modules to extract from

| File | What to Extract | Target |
|------|-----------------|--------|
| `generation_test_v2.py` lines 48-166 | `V3Cache(DynamicCache)` | `cache.py` — generalized, model-agnostic cache |

## 3. Architecture

### 3.1 Project Structure

```
TurboQuant-Research/
  turboquant/                     # Core library (renamed from turboquant-pytorch-master/)
    __init__.py
    lloyd_max.py                  # + disk-cache wrapper for codebooks
    turboquant.py                 # unchanged
    compressors.py                # unchanged
    compressors_v3.py             # + layer_bits param, + outlier_mask param
    cache.py                      # NEW: model-agnostic compressed cache
    adaptive.py                   # NEW: sensitivity profiling + DP bit allocation
    outlier.py                    # NEW: outlier channel detection + hybrid compression
    baselines/
      __init__.py
      kivi.py                     # KIVI (ICML 2024) baseline
      polarquant.py               # PolarQuant baseline
      fp16.py                     # Passthrough baseline
      registry.py                 # Method name -> cache factory mapping
  eval/
    __init__.py
    runner.py                     # Orchestrates: load model -> create cache -> run benchmark -> save JSON
    perplexity.py                 # WikiText-2, C4 sliding-window perplexity
    downstream.py                 # lm-evaluation-harness wrapper (MMLU, ARC, HellaSwag, WinoGrande, GSM8K)
    needle.py                     # Needle-in-haystack across context lengths and positions
    longbench.py                  # LongBench-E multi-task long-context evaluation
    metrics.py                    # Latency (tokens/sec), peak memory, compression ratio
    model_loader.py               # Multi-model loader (FP16/BF16, no weight quantization)
  configs/
    models/                       # YAML per model (name, dtype, max_seq_len)
    methods/                      # YAML per method (type, bit configs, window size)
    sweeps/
      main-paper.yaml             # Full experiment matrix
      quick-test.yaml             # Single-model smoke test
  scripts/
    run_experiment.py             # CLI: --config <sweep.yaml> [--model X] [--method Y] [--benchmark Z]
    aggregate_results.py          # Collect results/ JSONs -> LaTeX tables + CSV
  results/                        # gitignored, JSON per experiment run
  docs/superpowers/specs/         # This spec
```

### 3.2 Data Flow

```
Config YAML
    |
    v
runner.py ──> model_loader.py ──> HuggingFace model (FP16/BF16, device_map="auto")
    |                                    |
    v                                    v
registry.py ──> cache factory ──> CompressedCache(DynamicCache)
    |                                    |
    v                                    v
benchmark module ──> model.generate() / model() with past_key_values=cache
    |
    v
JSON result file ──> aggregate_results.py ──> LaTeX tables
```

### 3.3 Key Interface: CompressedCache

All methods (TurboQuant V3, Adaptive, Outlier, KIVI, PolarQuant, FP16) implement the same interface by subclassing HuggingFace's `DynamicCache`:

```python
class CompressedCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Accept new KV, compress old tokens, return full (decompressed + recent) KV."""
        # Returns: (full_keys, full_values) as tensors
    
    def get_seq_length(self, layer_idx=0):
        """Total sequence length across compressed + fp16 tokens."""
    
    def get_compression_info(self):
        """Report compressed vs fp16 token counts."""
```

This is the exact pattern already working in `generation_test_v2.py:V3Cache`. All baselines follow this interface so the evaluation code is method-agnostic.

## 4. Component Specifications

### 4.1 `cache.py` — Generalized Compressed Cache

**Extracted from:** `generation_test_v2.py` lines 48-166

**Changes from original:**
- Constructor accepts `model_config` (HuggingFace config object) to auto-detect `n_layers` via `model_config.num_hidden_layers` and `head_dim` via `model_config.head_dim` or `model_config.hidden_size // model_config.num_attention_heads`
- Constructor accepts `compressor_factory: Callable[[int, int, str], Compressor]` — a function that creates a compressor given `(layer_idx, head_dim, device)`. This allows swapping TurboQuant V3, Adaptive, Outlier, KIVI, or PolarQuant without changing cache logic.
- The incremental compression logic (chunk tracking, overflow detection, decompress-concat) is preserved exactly from the original `V3Cache.update()` method.
- Removes hardcoded `n_layers=36`

### 4.2 `adaptive.py` — Sensitivity-Adaptive Bit Allocation

Two functions:

**`profile_layer_sensitivity(model, tokenizer, n_samples=128, bit_options=[2,3,4,6,8]) -> Dict[int, Dict[int, float]]`**
- Loads `n_samples` from C4 validation split, tokenizes to `max_seq_len` tokens each
- **Phase 1 (capture):** Run a single FP16 forward pass with hooks on each decoder layer to capture K/V tensors and record the baseline logits. This is one forward pass total, not per-layer.
- **Phase 2 (simulate):** For each layer L and each bit-width B in `bit_options`:
  - Create a temporary `MSECompressor` with bit-width B
  - Compress and decompress the captured K/V for layer L only
  - Run a second forward pass where layer L receives the quantized K/V and all other layers receive their original FP16 K/V. This is done by injecting the quantized K/V via hooks, not by running partial model slices.
  - Compute KL divergence between the resulting logits and the baseline logits
- **Cost:** `n_layers * len(bit_options)` forward passes per sample. For 32 layers and 5 bit options, this is 160 passes per sample. With `n_samples=128` this is expensive. **Optimization:** Use only 8-16 samples for profiling (sufficient for stable sensitivity estimates), and use a shorter `max_seq_len` (512 tokens) during calibration.
- Returns `{layer_idx: {bits: kl_divergence}}` — the sensitivity curve

**`calibrate(model, tokenizer, n_samples=16, max_seq_len=512) -> Dict[int, Dict[str, torch.Tensor]]`**
- Shared calibration function used by both `profile_layer_sensitivity` and `OutlierProfiler`
- Returns captured K/V tensors per layer: `{layer_idx: {"keys": Tensor, "values": Tensor}}`
- Tensors shape: `(n_samples, n_heads, seq_len, head_dim)`

**`allocate_bits(sensitivity: Dict, budget: float, bit_options=[2,3,4,6,8]) -> Dict[int, Tuple[int,int]]`**
- Solves: minimize `sum(sensitivity[l][b_k] + sensitivity[l][b_v])` subject to `sum(b_k + b_v) / (2 * n_layers) <= budget`
- Uses dynamic programming: state = (layer_index, remaining_budget), choices = all (key_bits, value_bits) pairs from `bit_options`
- Returns `{layer_idx: (key_bits, value_bits)}`

**Integration:** `TurboQuantV3.__init__` gains an optional `layer_bits: Optional[Tuple[int,int]] = None` parameter. When provided, it overrides the protected/unprotected binary logic (current lines 158-159 of `compressors_v3.py`) with the given (key_bits, value_bits) for that specific layer.

### 4.3 `outlier.py` — Outlier-Aware Channel Grouping

Two classes:

**`OutlierProfiler`**
- `profile(model, tokenizer, n_samples=128) -> Dict[int, torch.BoolTensor]`
- Calls `calibrate()` from `adaptive.py` (shared calibration function) to get captured K/V tensors
- Computes per-channel magnitude statistics across all captured K/V tensors at each layer
- Marks channels where `channel_mean + 2 * channel_std > 5 * median_magnitude` as outliers
- Returns `{layer_idx: outlier_mask}` where `outlier_mask` has shape `[head_dim]`
- Expected: ~2-5% of channels flagged as outliers in Llama models

**`OutlierAwareMSECompressor`**
- Wraps `MSECompressor` from `compressors_v3.py`
- `compress(states, outlier_mask)`: splits states into outlier channels (stored as FP16) and normal channels (quantized by inner `MSECompressor` with `head_dim` reduced to non-outlier count)
- `decompress(compressed)`: reconstructs by merging FP16 outlier channels back with dequantized normal channels
- `memory_bytes()`: accounts for both FP16 outlier storage and quantized normal storage

**Integration with TurboQuantV3:** `TurboQuantV3.__init__` gains optional `outlier_masks: Optional[Dict[str, torch.BoolTensor]] = None` (keys: "key", "value"). When provided, uses `OutlierAwareMSECompressor` instead of plain `MSECompressor`.

### 4.4 `baselines/kivi.py` — KIVI Baseline

**`KIVICompressor`**
- Per-channel asymmetric quantization with configurable `group_size` (default 128)
- Keys: quantized per-channel along `head_dim` dimension
- Values: quantized per-token
- Quantize: `scale = (max - min) / (2^bits - 1)`, `zero_point = min`, `q = round((x - zero_point) / scale)`, stored as `uint8`
- Dequantize: `x_hat = q * scale + zero_point`

**`KIVICache(DynamicCache)`**
- Same incremental update pattern as `cache.py` — compress overflow beyond residual window
- Uses `KIVICompressor` instead of `MSECompressor`

### 4.5 `baselines/polarquant.py` — PolarQuant Baseline

**`PolarQuantCompressor`**
- Inherits structure from `MSECompressor` but replaces the random rotation matrix with a data-dependent rotation derived from the polar decomposition of the model's key/value projection weight matrices
- Requires access to `model.layers[i].self_attn.k_proj.weight` at init time
- Everything else (Lloyd-Max quantization, bit-packing, norms) is identical to `MSECompressor`

**`PolarQuantCache(DynamicCache)`**
- Same cache pattern, uses `PolarQuantCompressor`

### 4.6 `baselines/registry.py` — Method Registry

```python
METHODS = {
    "fp16":                     lambda cfg, model_cfg: None,  # no custom cache = default DynamicCache
    "turboquant-v3":            lambda cfg, model_cfg: TurboQuantCache(cfg, model_cfg),
    "turboquant-adaptive":      lambda cfg, model_cfg: TurboQuantCache(cfg, model_cfg, adaptive=True),
    "turboquant-outlier":       lambda cfg, model_cfg: TurboQuantCache(cfg, model_cfg, outlier=True),
    "turboquant-adaptive-outlier": lambda cfg, model_cfg: TurboQuantCache(cfg, model_cfg, adaptive=True, outlier=True),
    "kivi":                     lambda cfg, model_cfg: KIVICache(cfg, model_cfg),
    "polarquant":               lambda cfg, model_cfg: PolarQuantCache(cfg, model_cfg),
}
```

### 4.7 `eval/perplexity.py`

**`evaluate_perplexity(model, tokenizer, cache_factory, dataset="wikitext2", max_seq_len=2048, stride=None) -> dict`**
- Loads WikiText-2-raw-v1 or C4-en validation via `datasets` library
- Concatenates all text into a single token sequence
- Processes in chunks of `max_seq_len` with `stride = max_seq_len // 2` (default)
- For each chunk: creates fresh cache via `cache_factory()`, runs `model(input_ids, past_key_values=cache, use_cache=True)`, accumulates cross-entropy loss on non-overlapping portion
- Returns `{"perplexity": float, "loss": float, "n_tokens": int}`

**Critical detail:** For perplexity, we do NOT use `model.generate()`. We use the forward pass directly with `labels` to get loss, or manually compute cross-entropy from logits. A fresh cache is created per sliding window chunk (not per document) to simulate the real-world pattern where the KV cache accumulates tokens within a context window. This also prevents unbounded memory growth.

### 4.8 `eval/downstream.py`

**`CompressedCacheLM(lm_eval.api.model.LM)`**
- Wraps a HuggingFace model + compressed cache factory
- Implements `loglikelihood(requests)`: for each (context, continuation) pair, creates cache, runs forward pass, extracts log-probs of continuation tokens
- Implements `generate_until(requests)`: for each (context, stop_criteria) pair, creates cache, calls `model.generate()` with cache
- Implements `loglikelihood_rolling(requests)`: sliding-window log-likelihood with cache

**`evaluate_downstream(model, tokenizer, cache_factory, tasks, num_fewshot) -> dict`**
- Calls `lm_eval.evaluator.simple_evaluate(model=CompressedCacheLM(...), tasks=tasks, num_fewshot=num_fewshot)`
- Returns per-task accuracy/score dict

**Fallback:** If `lm_eval` integration proves too complex, `evaluate_downstream_manual()` computes MMLU/ARC log-likelihoods directly: for each multiple-choice question, compute log-prob of each answer choice conditioned on the prompt, pick the highest.

### 4.9 `eval/needle.py`

**`evaluate_needle(model, tokenizer, cache_factory, context_lengths, needle_positions) -> dict`**
- Refactored from `generation_test_v2.py:run_test()`
- For each (context_length, needle_position) pair: builds prompt with filler text + hidden needle, generates with compressed cache, checks EXACT/PARTIAL/MISS
- Returns 2D result grid: `{(ctx_len, pos): "EXACT"|"PARTIAL"|"MISS"}`

### 4.10 `eval/metrics.py`

**`measure_latency(model, tokenizer, cache_factory, prompt_len=2048, gen_len=128) -> dict`**
- Measures prefill time (forward pass on full prompt) and decode time (per-token generation)
- Returns `{"prefill_ms": float, "decode_tokens_per_sec": float, "total_ms": float}`

**`measure_memory(model, cache_factory, seq_len=4096) -> dict`**
- `torch.cuda.reset_peak_memory_stats()` before, `torch.cuda.max_memory_allocated()` after
- Returns `{"peak_memory_mb": float, "cache_memory_mb": float, "model_memory_mb": float}`

### 4.11 `eval/model_loader.py`

**`load_model(model_name, dtype="auto", device_map="auto") -> (model, tokenizer)`**
- `dtype="auto"` selects BF16 for Llama-3.x/Mistral, FP16 otherwise
- No bitsandbytes weight quantization — 144GB VRAM handles up to 70B in FP16
- Returns `(model, tokenizer)` with `model.eval()` already called

### 4.12 `lloyd_max.py` — Codebook Disk Cache

**Modification to `LloydMaxCodebook.__init__`:**
- Before computing, check for cached `.pt` file at `~/.cache/turboquant/codebook_{d}_{bits}.pt`
- If found, load and return
- If not, compute via existing `solve_lloyd_max()`, save to cache, return
- Cache key: `(d, bits)` — the only two parameters that affect the codebook

## 5. Configuration System

### 5.1 Model Config (`configs/models/llama-3.1-8b.yaml`)

```yaml
name: meta-llama/Llama-3.1-8B-Instruct
dtype: bfloat16
max_seq_len: 8192
```

### 5.2 Method Config (`configs/methods/turboquant-v3.yaml`)

```yaml
type: turboquant-v3
params:
  key_bits: 4
  value_bits: 4
  residual_window: 128
  protected_layers: 0
```

### 5.3 Sweep Config (`configs/sweeps/main-paper.yaml`)

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
    params: {budget: 4.0, residual_window: 128, calibration_samples: 128}
  - type: turboquant-outlier
    params: {key_bits: 4, value_bits: 4, residual_window: 128, calibration_samples: 128}
  - type: turboquant-adaptive-outlier
    params: {budget: 4.0, residual_window: 128, calibration_samples: 128}
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

### 5.4 Result JSON Schema

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "method": "turboquant-adaptive",
  "method_config": {"budget": 4.0, "residual_window": 128},
  "benchmark": "perplexity",
  "benchmark_config": {"dataset": "wikitext2", "max_seq_len": 2048},
  "results": {"perplexity": 5.42, "loss": 1.69, "n_tokens": 245567},
  "hardware_metrics": {"peak_memory_mb": 18432, "prefill_ms": 342},
  "timestamp": "2026-04-07T14:32:00Z",
  "git_sha": "abc123"
}
```

## 6. Experiment Matrix

### 6.1 Models

| Model | Parameters | Head Dim | Layers | Max Seq | Priority |
|-------|-----------|----------|--------|---------|----------|
| Llama-3.2-3B-Instruct | 3.2B | 128 | 28 | 131K | High |
| Llama-3.1-8B-Instruct | 8B | 128 | 32 | 128K | High (primary, matches paper) |
| Mistral-7B-Instruct-v0.3 | 7B | 128 | 32 | 32K | High |
| Llama-3.1-70B-Instruct | 70B | 128 | 80 | 128K | Medium (if compute allows) |

### 6.2 Methods

| Method | Type | Description |
|--------|------|-------------|
| FP16 | Baseline | No compression (upper bound) |
| KIVI-4 | Baseline | Per-channel 4-bit asymmetric quant, group_size=128 |
| KIVI-2 | Baseline | Per-channel 2-bit, aggressive compression |
| PolarQuant-4 | Baseline | Polar decomposition + Lloyd-Max, 4-bit |
| TQ-V3-K4V4 | Replication | TurboQuant V3 with 4-bit keys, 4-bit values |
| TQ-V3-K6V4 | Replication | TurboQuant V3 with 6-bit keys, 4-bit values |
| TQ-Adaptive | Novel (A) | Sensitivity-adaptive per-layer bit allocation, avg 4 bits |
| TQ-Outlier | Novel (B) | Outlier-aware channel grouping + 4-bit quantization |
| TQ-Adaptive+Outlier | Novel (A+B) | Both techniques combined |

All TQ methods use `residual_window=128` unless sweeping window size.

### 6.3 Benchmarks

| Benchmark | Metrics | Purpose |
|-----------|---------|---------|
| WikiText-2 perplexity | PPL, PPL delta | Primary quality metric |
| C4 perplexity | PPL, PPL delta | Generalization check |
| MMLU (5-shot) | Accuracy | Knowledge retention |
| ARC-Challenge (25-shot) | Accuracy | Reasoning |
| HellaSwag (10-shot) | Accuracy | Common sense |
| WinoGrande (5-shot) | Accuracy | Coreference |
| GSM8K (5-shot CoT) | Accuracy | Math reasoning |
| Needle-in-haystack | EXACT/PARTIAL/MISS heatmap | Long-context retrieval |
| LongBench-E (8B only) | Task-specific scores | Long-context understanding |

### 6.4 Ablation Studies

1. **Adaptive allocation visualization:** Per-layer bit assignment heatmap across models (do different architectures allocate bits differently?)
2. **Outlier channel analysis:** Which layers/channels are flagged as outliers? How does this correlate with model architecture?
3. **Component ablation at 4-bit budget:** FP16 vs TQ-V3 vs TQ-Adaptive vs TQ-Outlier vs TQ-Adaptive+Outlier (isolate each contribution)
4. **Residual window sweep:** rw={0, 64, 128, 256, 512} at K4V4 across all models
5. **Bit budget sweep:** avg {2, 3, 4, 6} bits with adaptive allocation vs uniform allocation

### 6.5 Run Order

1. Llama-3.1-8B: all methods, all benchmarks (most comparable to paper, establishes baselines)
2. Mistral-7B: all methods, all benchmarks (validates generalization across architectures)
3. Llama-3.2-3B: all methods, all benchmarks (small model behavior)
4. Llama-3.1-70B: FP16 + best TQ configs only, PPL + MMLU (scale validation)

## 7. Fallback Strategy

If sensitivity-adaptive allocation (Contribution A) shows <0.1 PPL improvement over uniform allocation:
- The paper leads with outlier-aware channel grouping (Contribution B) as the primary contribution
- Adaptive allocation is reported as a negative result with analysis of why per-layer sensitivity is insufficient

If BOTH contributions show <0.1 PPL improvement:
- Pivot to a strong empirical study: "Benchmarking KV Cache Compression Across Model Scales"
- The QJL-hurts-for-softmax finding (confirmed by 6+ teams, documented in README) becomes a key insight
- Focus on when/why TurboQuant wins or loses vs KIVI/PolarQuant at different scales
- Target workshops rather than main conference

If both work: paper presents them as complementary with a full ablation study.

## 8. Dependencies

```
# Existing
torch>=2.0.0
scipy>=1.10.0
transformers>=4.40.0
accelerate>=0.25.0

# Remove (don't use weight quantization for benchmarks)
# bitsandbytes>=0.43.0  -- only needed if running on <24GB GPU

# New
lm-eval>=0.4.0              # EleutherAI evaluation harness
datasets>=2.14.0             # HuggingFace datasets
pyyaml>=6.0                  # Config files
rouge-score>=0.1.2           # LongBench scoring
pandas>=2.0                  # Results aggregation
matplotlib>=3.7              # Paper figures
seaborn>=0.13                # Heatmaps
```

## 9. Hardware

Server: `asus@140.113.202.36`
- 144GB VRAM (sufficient for 70B model in FP16 at ~140GB)
- `device_map="auto"` for multi-GPU distribution

## 10. Success Criteria

1. **Replication**: TQ-V3 PPL on Llama-3.1-8B at 4-bit matches or approaches the TurboQuant paper's reported quality-neutrality at 3.5-bit
2. **Novel contribution**: At least one of {Adaptive, Outlier} shows statistically significant improvement (>0.1 PPL or >0.5% downstream accuracy) over uniform TQ-V3 at the same average bit budget
3. **Completeness**: All cells in the experiment matrix are filled, results are reproducible via config + script
4. **Paper-ready**: LaTeX tables generated automatically from result JSONs
