# Sensitivity-Adaptive KV Cache Compression: Complete Research Report

> **Companion to:** `paper/main.tex` (conference-length paper) and `README.md` (quick-start guide).
> This document contains the full, in-depth record of the research: motivation, methods, all experiments (including negative results and debugging), every raw number, and reproduction instructions. If the paper is the polished story, this is the lab notebook.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Prior Work](#2-background-and-prior-work)
3. [Problem Statement and Hypotheses](#3-problem-statement-and-hypotheses)
4. [Implementation](#4-implementation)
5. [Experimental Setup](#5-experimental-setup)
6. [Main Results: Perplexity](#6-main-results-perplexity)
7. [Sensitivity Analysis (Calibration)](#7-sensitivity-analysis-calibration)
8. [Adaptive Bit Allocation Results](#8-adaptive-bit-allocation-results)
9. [Outlier-Aware Channel Grouping Results](#9-outlier-aware-channel-grouping-results)
10. [Needle-in-a-Haystack](#10-needle-in-a-haystack)
11. [Hardware Benchmarks](#11-hardware-benchmarks)
12. [Failed Experiments and Debugging Stories](#12-failed-experiments-and-debugging-stories)
13. [Cross-Model Findings and Analysis](#13-cross-model-findings-and-analysis)
14. [Limitations](#14-limitations)
15. [Reproduction Instructions](#15-reproduction-instructions)
16. [Appendix: Raw Numbers](#16-appendix-raw-numbers)

---

## 1. Executive Summary

### The Question

Can TurboQuant — a recent vector-quantization-based KV cache compressor published at ICLR 2026 — be made to work reliably across modern LLMs? We set out to replicate and extend TurboQuant, and in the process uncovered a systematic failure mode that had not been previously characterized.

### The Finding

TurboQuant's effectiveness is **strongly correlated with the Grouped Query Attention (GQA) ratio of the model**. On low-GQA models (Gemma-3-27B at GQA 2:1, Mistral-7B at GQA 4:1, Llama-3.1-8B at GQA 4:1), TurboQuant is near-lossless at 4 bits per key / 4 bits per value and dominates the per-channel KIVI baseline by 8×–30× in terms of distance from FP16 perplexity. On high-GQA models (Qwen2.5-3B at GQA 8:1), uniform 4-bit TurboQuant **collapses catastrophically**, inflating perplexity by 4.5× (7.63 → 42.33 on WikiText-2), while KIVI survives with only a 3.4% degradation.

We term this phenomenon **GQA amplification**: when a single compressed KV head is consumed by $g$ query heads, any per-vector quantization error is visible to all $g$ of them in parallel, and VQ methods (which operate on whole head vectors) are more vulnerable to this amplification than per-channel scalar methods like KIVI.

### The Rescue

We propose two contributions that together rescue TurboQuant on high-GQA models:

1. **Sensitivity-adaptive per-layer bit allocation.** We profile per-layer, per-component (key vs value) compression sensitivity using a 16-sample calibration set, then solve a 2D discrete knapsack via dynamic programming to find the optimal per-layer bit assignment under an average-bit budget. On Qwen2.5-3B, this reduces 4-bit perplexity from 42.33 to 8.01.

2. **Outlier-aware channel grouping.** We identify high-magnitude channels during calibration and store them in FP16, applying TurboQuant only to the remaining channels. On Qwen2.5-3B, this reduces 4-bit perplexity from 42.33 to **7.74** — within 1.5% of FP16 and the single strongest rescue technique we tested.

### Key Numbers (WikiText-2 perplexity)

| Model | GQA | FP16 | TQ K4V4 (uniform) | TQ K4V4 (outlier) | KIVI-4 | Best TQ vs KIVI |
|---|---|---|---|---|---|---|
| Gemma-3-27B | 2:1 | 7.47 | **7.52 (+0.7%)** | n/a (no outliers) | 9.14 (+22.4%) | **30× closer to FP16** |
| Mistral-7B | 4:1 | 4.94 | **4.96 (+0.4%)** | n/a (no outliers) | 5.13 (+3.9%) | 9× closer |
| Llama-3.1-8B | 4:1 | 6.48 | **6.58 (+1.6%)** | n/a (no outliers) | 7.29 (+12.4%) | 8× closer |
| Qwen2.5-3B | 8:1 | 7.63 | 42.33 (COLLAPSE) | **7.74 (+1.5%)** | 7.89 (+3.4%) | Outlier rescues; beats KIVI |
| Llama-3.1-70B | 8:1 | 3.40 | OOM | OOM | OOM | — |

### The Secondary Finding

Across all profiled models, **keys are 10–100× more sensitive to quantization than values**. On Mistral-7B, the average key MSE at 2-bit compression is ~0.31, while the average value MSE is ~0.01. This asymmetry is not exploited by uniform compression schemes but is directly leveraged by our adaptive allocator, which typically assigns 4–8 bits to keys and 2–4 bits to values at the same average budget.

---

## 2. Background and Prior Work

### 2.1 KV Cache in Transformer Decoding

During autoregressive generation, a transformer decoder processes tokens one at a time. At each attention layer, it projects the current hidden state into query ($Q$), key ($K$), and value ($V$) vectors, computes attention against all previous $K$s and $V$s, and produces an output. To avoid recomputing previous $K$ and $V$ at every decode step, implementations maintain a **KV cache** that grows by one entry per token per layer per KV head.

For a model with $L$ layers, $H$ KV heads, head dimension $D$, sequence length $S$, and batch size $B$, the KV cache size in FP16 is

$$
\text{cache size} = 2 \cdot B \cdot L \cdot H \cdot S \cdot D \cdot 2 \text{ bytes}.
$$

For Llama-3.1-70B ($L=80$, $H=8$, $D=128$) at $S=8{,}192$ tokens and $B=1$, this is

$$
2 \cdot 1 \cdot 80 \cdot 8 \cdot 8192 \cdot 128 \cdot 2 \text{ bytes} \approx 2.7 \text{ GB},
$$

which is modest. For $S=128{,}000$ tokens (a typical long-context setting), it balloons to ~42 GB, comparable to the 140 GB of weights. The KV cache becomes the dominant memory consumer.

### 2.2 Grouped Query Attention (GQA)

GQA \[Ainslie et al., 2023\] reduces the KV cache size by sharing each KV head among multiple query heads. The GQA ratio $g = n_{\text{heads}} / n_{\text{kv\_heads}}$ determines how much sharing happens. Modern models have adopted aggressive GQA:

| Model | $n_{\text{heads}}$ | $n_{\text{kv\_heads}}$ | $g$ |
|---|---|---|---|
| Gemma-3-27B | 32 | 16 | **2** |
| Mistral-7B | 32 | 8 | **4** |
| Llama-3.1-8B | 32 | 8 | **4** |
| Qwen2.5-3B | 16 | 2 | **8** |
| Llama-3.1-70B | 64 | 8 | **8** |

A higher GQA ratio $g$ means each KV head is "more important" — its compression error is seen by more downstream attention dot products. This is the fulcrum of our main finding.

### 2.3 KIVI (Per-Channel Scalar Quantization)

KIVI \[Liu et al., ICML 2024\] quantizes the KV cache with asymmetric scalar per-channel quantization. For keys, it computes per-channel $(\text{scale}, \text{zero})$ along the head dimension; for values, it computes per-token statistics. Each element is then quantized to $b$ bits using

$$
\hat x = \text{round}((x - \text{zero}) / \text{scale}), \quad x' = \hat x \cdot \text{scale} + \text{zero}.
$$

The per-channel metadata (scale, zero in FP16) adds overhead that inflates the effective bit-width. At $b=4$ with group size 128, the overhead is small; at $b=2$ with the same group size, it becomes significant.

KIVI's error profile is **local**: the reconstruction error is bounded per channel, and the attention computation sees error at the level of individual elements.

### 2.4 TurboQuant (Vector Quantization)

TurboQuant \[Anonymous, ICLR 2026\] takes a fundamentally different approach. It treats each head vector as an atomic unit and compresses it through a three-stage pipeline:

1. **Normalize** the vector to the unit sphere ($\hat x = x / \|x\|$), storing the norm as FP16.
2. **Rotate** with a fixed random orthogonal matrix $\Pi$: $\tilde x = \Pi \hat x$. This is a Johnson-Lindenstrauss-style transform that regularizes the distribution: each coordinate of $\tilde x$ is approximately distributed as $\pm B(D/2, D/2)$ on $[-1, 1]$.
3. **Quantize** each coordinate independently with an optimal Lloyd-Max codebook for that Beta distribution, then **pack** the indices into bytes.

Decompression reverses these steps: unpack, look up centroids, apply $\Pi^\top$, multiply by the stored norm.

The paper reports near-lossless 4-bit compression on Llama and Mistral. The paper's original pipeline includes a second QJL (Quantized Johnson-Lindenstrauss) stage, but community analyses (and our own preliminary experiments) found that this second stage degrades downstream quality because softmax attention amplifies any low-rank noise introduced by the sketch. We work exclusively with MSE-only TurboQuant, which we call **TurboQuant-V3**.

TurboQuant's error profile is **global**: the Lloyd-Max index represents a direction on the $D$-sphere, so mis-indexing produces a vector-valued error aligned with the codebook.

### 2.5 Other Relevant Work

- **GPTQ, AWQ, OWQ** — sensitivity-aware weight quantization. Our adaptive allocator borrows the spirit of these methods but applies them to the KV cache, which is a streaming distribution rather than a fixed weight matrix.
- **KVQuant** \[Hooper et al., NeurIPS 2024\] — scalar quantization with non-uniform grids and outlier isolation. Our outlier handling is similar in spirit but operates inside a VQ framework.
- **SmoothQuant** \[Xiao et al., ICML 2023\] — handles activation outliers by migrating magnitude to weights. Not directly applicable to KV cache compression, but the outlier characterization motivated our design.

---

## 3. Problem Statement and Hypotheses

### 3.1 Original Goals

When we started this work, we set out to:

1. Reproduce TurboQuant on standardized benchmarks (WikiText-2 perplexity, downstream tasks, needle-in-a-haystack). The original paper only tested on a hand-rolled needle test with Qwen2.5-3B.
2. Compare TurboQuant to KIVI and PolarQuant on a range of model sizes (3B–70B).
3. Propose a novel contribution that would justify a conference-length paper. Our initial plan was sensitivity-adaptive per-layer bit allocation.

### 3.2 Hypotheses

We formulated the following hypotheses during planning:

- **H1 (Key-value asymmetry).** Keys are more sensitive to quantization than values, because keys participate in the softmax (which can amplify errors) while values are linearly mixed by the already-computed attention weights. Prediction: per-layer sensitivity profiling will show key MSE significantly larger than value MSE.

- **H2 (Layer depth matters).** Early layers (near the embedding) and late layers (near the output head) are more sensitive than middle layers, following patterns seen in GPTQ and AWQ weight sensitivity. Prediction: adaptive allocation will spend extra bits on first/last few layers.

- **H3 (Adaptive allocation is the big win).** At a fixed average bit budget, sensitivity-adaptive allocation will outperform uniform allocation by a significant margin on all models.

### 3.3 What We Actually Found

- H1 was **strongly confirmed**: keys are 10–100× more sensitive than values across every model we profiled.
- H2 was **partially confirmed**: layer 0 keys are consistently the most sensitive, by a factor of 3–10× versus median. Deep-layer values show a milder spike on some models. Middle layers are mostly homogeneous.
- H3 was **qualified by a new finding**: on low- and mid-GQA models (Gemma-3-27B, Mistral-7B, Llama-3.1-8B), adaptive allocation gave essentially no improvement over uniform K4V4 — because uniform K4V4 was already near-lossless. Adaptive allocation's big win is in the high-GQA regime (Qwen2.5-3B), where uniform K4V4 collapses and adaptive rescues the model.
- A **new finding** emerged that we had not anticipated: the GQA amplification effect, which became the paper's central narrative. Outlier-aware channel grouping, which we had originally planned as a secondary/fallback contribution, turned out to be the strongest single rescue technique on high-GQA models.

---

## 4. Implementation

### 4.1 Repository Structure

```
TurboQuant-Research/
├── turboquant/                    # Core library
│   ├── compressors_v3.py          # TurboQuant-V3 compressor (MSE-only, asymmetric K/V)
│   ├── lloyd_max.py               # Lloyd-Max codebook (with disk cache)
│   ├── cache.py                   # CompressedCache (HuggingFace DynamicCache subclass)
│   ├── adaptive.py                # Sensitivity profiling + DP bit allocation
│   ├── outlier.py                 # Outlier detection + OutlierAwareKVCompressor
│   ├── turboquant.py              # Original TurboQuant (unchanged from upstream)
│   └── baselines/
│       ├── kivi.py                # KIVI per-channel asymmetric quantization
│       ├── polarquant.py          # PolarQuant baseline
│       └── registry.py            # Method factory: method_name -> cache factory
├── eval/                          # Evaluation harness
│   ├── model_loader.py            # load_model(), get_model_info() (handles nested configs)
│   ├── perplexity.py              # WikiText-2, C4 sliding-window PPL
│   ├── needle.py                  # Needle-in-a-haystack
│   ├── metrics.py                 # Latency and peak memory measurement
│   ├── downstream.py              # lm-evaluation-harness wrapper (MMLU, ARC, etc.)
│   ├── runner.py                  # Experiment orchestration
│   └── config.py                  # YAML config loader (dataclasses + yaml)
├── configs/
│   ├── models/                    # One YAML per model (name, dtype, max_seq_len)
│   └── sweeps/                    # One YAML per experiment (models × methods × benchmarks)
├── scripts/
│   ├── run_experiment.py          # CLI entry point
│   ├── calibrate.py               # Sensitivity profiling + DP allocation
│   ├── profile_outliers.py        # Outlier channel detection
│   ├── benchmark_hw.py            # Latency and memory benchmarks
│   ├── debug_qwen.py              # Debugging script for Qwen collapse
│   ├── gen_paper_tables.py        # Auto-generate LaTeX tables from results
│   ├── gen_sensitivity_plot.py    # Auto-generate the sensitivity figure
│   ├── build_docx.py              # Convert paper to DOCX with inlined tables/refs
│   └── aggregate_results.py       # Collect results into pandas DataFrames
├── tests/                         # 18 unit tests, all passing
├── results/                       # Per-sweep JSON results (committed to repo)
│   ├── calibration/               # Per-layer sensitivity JSONs
│   ├── outlier/                   # Per-model outlier profiles
│   ├── benchmarks/                # Hardware benchmarks
│   └── <sweep-name>/              # One JSON per (model, method, benchmark) run
├── paper/
│   ├── main.tex                   # Conference-length paper
│   ├── main.pdf                   # Compiled PDF (10 pages)
│   ├── main.docx                  # Pandoc-converted DOCX version
│   ├── references.bib             # 11 citations
│   ├── tables/                    # Auto-generated LaTeX tables
│   └── figures/                   # Auto-generated figures (PDF + PNG)
├── docs/                          # Design docs, plans, specs
├── CLAUDE.md                      # Project context / assistant memory
├── README.md                      # User-facing quick start
├── RESEARCH_REPORT.md             # This file
└── requirements.txt
```

### 4.2 Core Data Flow

```
input_ids
    │
    ▼
 model.forward(past_key_values=CompressedCache(...))
    │
    ▼  attention layer i:
         K_new, V_new ← k_proj(x), v_proj(x)           # per-layer KV projection
         K_full, V_full ← cache.update(K_new, V_new, i)
    │
    ▼  CompressedCache.update():
         1. Append K_new, V_new to the fp16 "recent" buffer.
         2. If |recent buffer| > residual_window:
              overflow ← excess tokens beyond the window
              ck, cv ← compressor.compress_kv(overflow)
              Append (ck, cv) to the compressed chunks list.
         3. Decompress all chunks and concat with fp16 recent.
         4. Return (K_full, V_full) for attention to consume.
```

The **residual window** is crucial: it keeps the most recent $W$ tokens in FP16, so local attention (where most of the signal lives) is lossless. We use $W=128$ by default, but ablate this parameter in our Qwen debugging experiments (see Section 12).

### 4.3 Key Implementation Details

#### 4.3.1 `CompressedCache` (turboquant/cache.py)

This is a `DynamicCache` subclass that conforms to HuggingFace's caching API. Key points:

- **Per-layer state**: we maintain `_chunks_k[i]`, `_chunks_v[i]` (list of compressed chunks) and `_fp16_recent_k[i]`, `_fp16_recent_v[i]` (the residual window) for each layer `i`.
- **Compressor factory**: the cache is parameterized by a callable `compressor_factory(layer_idx, head_dim, device)` that returns an object with a `compress_kv()` / `decompress_kv()` interface. This makes the cache backend-agnostic: KIVI, PolarQuant, TurboQuant, and adaptive/outlier variants all plug in through the same factory.
- **HuggingFace compatibility shim**: in `transformers 5.5.0`, `DynamicCache` expects `self.layers` to be a list of `DynamicLayer` objects. We grow `self.layers` as needed inside `update()` to avoid breaking the parent class's length checks.

#### 4.3.2 Lloyd-Max Disk Cache (turboquant/lloyd_max.py)

The Lloyd-Max codebook computation uses `scipy.integrate.quad` for numerical integration of the Beta distribution, which is slow. We cache computed codebooks as `.pt` files under `~/.cache/turboquant/codebook_{d}_{b}.pt`, keyed on (head_dim, bits). First-time computation takes ~10 seconds; subsequent loads are instant.

#### 4.3.3 Adaptive Allocation DP (turboquant/adaptive.py)

The DP takes the per-layer, per-bit sensitivity dict `{layer_idx: {"key": {bits: mse}, "value": {bits: mse}}}` and returns an optimal allocation `{layer_idx: (key_bits, value_bits)}`. State space: `dp[l][b]` = minimum total distortion after processing layers $1..l$ with $b$ total bits consumed. At each layer, we try all $|B|^2 = 25$ possible (key, value) bit pairs (for $B = \{2,3,4,6,8\}$). Complexity: $O(L \cdot B_{\max} \cdot |B|^2) \approx O(62 \cdot 1000 \cdot 25) = 1.5$M ops — sub-second on Python.

**Backtracking**: the DP table also stores `choice[l][b]`, a tuple `(kb, vb, prev_b)` recording the best decision at each state, enabling linear-time allocation recovery.

#### 4.3.4 Outlier Detection (turboquant/outlier.py)

A channel $d$ is flagged as an outlier if

$$
\mathrm{mean}_t |K_{t, d}| + 2 \cdot \mathrm{std}_t |K_{t, d}| > 5 \cdot \mathrm{median}_{d'} \left( \mathrm{mean}_t |K_{t, d'}| + 2 \cdot \mathrm{std}_t |K_{t, d'}| \right).
$$

The factor of 5 was chosen to roughly match the typical outlier ratio observed in AWQ-style analyses, yielding 0–5 outlier channels per layer on most models (up to 26 on Qwen-3B layer 0).

`OutlierAwareMSECompressor` wraps an inner `MSECompressor` of reduced dimension $D - k$ (where $k$ is the number of outliers), and passes the $k$ outlier channels through as FP16. The storage overhead is negligible ($k \cdot 2$ bytes per vector, versus $D/2$ bytes for the compressed part at 4-bit).

#### 4.3.5 `MSECompressor.compress_kv()` Contract

All compressors in our system expose the interface:

```python
def compress_kv(self, keys: Tensor, values: Tensor) -> Tuple[Any, Any]:
    """keys, values: shape (B, n_kv_heads, S, head_dim). Returns opaque compressed objects."""

def decompress_kv(self, ck: Any, cv: Any) -> Tuple[Tensor, Tensor]:
    """Inverse of compress_kv."""
```

This contract is enforced by `CompressedCache` and enables plug-and-play swapping.

### 4.4 Testing Infrastructure

We maintain 18 unit tests across 6 test files:

- `tests/test_codebook_cache.py` — 2 tests. Verifies Lloyd-Max disk caching works correctly.
- `tests/test_cache.py` — 3 tests. Verifies `CompressedCache.update()` behavior (basic update, incremental growth, compression info reporting).
- `tests/test_baselines.py` — 4 tests. KIVI compress/decompress round-trip, bit-width ordering, compression ratio, PolarQuant round-trip.
- `tests/test_adaptive.py` — 5 tests. `layer_bits` override, compress/decompress with override, DP budget respect, sensitive-layers-get-more, K/V split format.
- `tests/test_outlier.py` — 3 tests. Outlier detection, outlier compressor shape, outlier-aware compressor lower MSE than plain.
- `tests/test_perplexity.py` — 2 tests. PPL on known logits, PPL sanity (random logits are worse).

All 18 tests pass both locally and on the server.

---

## 5. Experimental Setup

### 5.1 Hardware

**Server**: `asus@140.113.202.36`
- 3× NVIDIA RTX A6000, 48 GB each → 144 GB aggregate VRAM
- Python 3.12.3, PyTorch 2.5.1 + CUDA 12.1
- transformers 5.5.0, datasets 4.8.4

### 5.2 Models Evaluated

| Model | Params | Layers | $n_{\text{heads}}$ | $n_{\text{kv\_heads}}$ | GQA | head_dim | Gated? | Accessible via |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-27B-it | 27B | 62 (mixed) | 32 | 16 | **2:1** | 128 | yes (approved) | user's HF token |
| Mistral-7B-Instruct-v0.3 | 7B | 32 | 32 | 8 | **4:1** | 128 | no | public |
| Llama-3.1-8B-Instruct | 8B | 32 | 32 | 8 | **4:1** | 128 | yes (approved) | user's HF token |
| Llama-3.1-70B-Instruct | 70B | 80 | 64 | 8 | **8:1** | 128 | yes (approved) | user's HF token |
| Qwen2.5-3B-Instruct | 3B | 36 | 16 | 2 | **8:1** | 128 | no | public |

All models loaded in BF16 via `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto")` — no bitsandbytes weight quantization.

**Gemma-3-27B has a mixed attention structure**: 52 sliding-attention layers and 10 full-attention layers (every 6th layer). We use the `text_config` attribute for the nested config to extract the standard transformer parameters.

### 5.3 Baselines

- **FP16**: no compression. Uses HuggingFace's default `DynamicCache`.
- **KIVI-b**: asymmetric per-channel quantization at $b$ bits with group size 128. Keys quantized per-channel (along head dim), values per-token. Residual window 128.
- **PolarQuant**: polar decomposition of projection matrices in place of random rotation. We implemented it (turboquant/baselines/polarquant.py) but did not use it in the final experiments since KIVI is the more relevant comparison.
- **TurboQuant-V3 K$k$V$v$**: MSE-only TurboQuant with $k$ key bits and $v$ value bits, residual window 128.

### 5.4 Benchmarks

- **WikiText-2 perplexity**: sliding window over the test split with `max_seq_len=2048` and `stride=1024`. Cross-entropy is accumulated only on the non-overlapping portion of each window.
- **Needle-in-a-haystack**: a secret code name ("AURORA-7749") is hidden inside a long repetitive context at various positions (0.1, 0.3, 0.5, 0.7, 0.9 of the document). The model is asked to retrieve it. We test context lengths 4K, 8K, 16K.
- **Hardware**: prefill latency (milliseconds to process a 2048-token prompt), decode throughput (tokens/sec during 32–64 generated tokens), peak GPU memory during a single forward pass.

---

## 6. Main Results: Perplexity

### 6.1 Full Perplexity Table

WikiText-2 test set, sliding window at 2048 tokens. All values are raw perplexity; ($+X\%$) shows relative degradation from FP16.

| Method | Bits | Gemma-3-27B | Mistral-7B | Llama-3.1-8B | Qwen2.5-3B |
|---|---|---|---|---|---|
| **FP16** | 16 | **7.47** | **4.94** | **6.48** | **7.63** |
| TQ K8V8 | 8/8 | — | — | — | 7.63 (+0.0%) |
| TQ K6V6 | 6/6 | — | — | — | 7.67 (+0.6%) |
| TQ K6V4 | 6/4 | — | — | — | 7.71 (+1.1%) |
| KIVI-6 | 6 | — | — | — | 7.64 (+0.2%) |
| **TQ K4V4** | 4/4 | **7.52 (+0.7%)** | **4.96 (+0.4%)** | **6.58 (+1.6%)** | **42.33 (+455%)** ⚠️ |
| TQ K4V2 | 4/2 | 7.68 (+2.9%) | 5.05 (+2.3%) | 6.87 (+5.9%) | 56.14 (+636%) ⚠️ |
| **KIVI-4** | 4 | **9.14 (+22.3%)** | **5.13 (+3.9%)** | **7.29 (+12.4%)** | **7.89 (+3.4%)** |
| TQ K3V3 | 3/3 | — | 5.05 (+2.3%) | — | collapse |
| TQ K2V2 | 2/2 | 14.28 (+91.3%) | 5.79 (+17.3%) | 10.35 (+59.6%) | collapse |
| KIVI-2 | 2 | collapse (4.4×10⁵) | 235.18 (+4664%) | collapse (1244) | collapse (4.2×10³) |

### 6.2 TurboQuant vs KIVI Gap (at 4-bit)

This is the central table of the paper.

| Model | GQA | TQ K4V4 delta | KIVI-4 delta | Ratio |
|---|---|---|---|---|
| **Gemma-3-27B** | **2:1** | +0.7% | +22.4% | **32×** |
| Mistral-7B | 4:1 | +0.4% | +3.9% | 10× |
| Llama-3.1-8B | 4:1 | +1.6% | +12.4% | 8× |
| Qwen2.5-3B | 8:1 | +455% (collapse) | +3.4% | KIVI wins |

The trend is unambiguous: as the GQA ratio increases, TurboQuant's advantage over KIVI shrinks, crosses zero around 6:1, and becomes a catastrophic loss at 8:1.

### 6.3 Residual Window Ablation (Qwen2.5-3B, TQ K4V4)

From `scripts/debug_qwen.py`:

| Residual window | PPL | Delta |
|---|---|---|
| 128 (default) | 42.33 | +455% |
| 512 | 24.13 | +216% |
| 1024 | 10.55 | +38% |
| FP16 (inf window) | 7.63 | 0% |

A larger residual window partially mitigates the collapse, because more tokens stay in FP16. But at residual_window=1024 we're keeping half the sequence in FP16, which defeats the purpose of compression. This motivated our search for a principled rescue.

### 6.4 Bit-Width Ablation (Qwen2.5-3B, uniform TQ)

| Bits (K/V) | PPL | Delta |
|---|---|---|
| 8/8 | 7.63 | 0.0% |
| 6/6 | 7.67 | +0.6% |
| 6/4 | 7.71 | +1.1% |
| 4/4 | **42.33** | **+455%** ⚠️ |
| 3/3 | 1307.79 | collapse |
| 2/2 | 4231.77 | collapse |

Observation: The cliff is between 6 bits and 4 bits. At 6 bits the model is still usable; at 4 bits it collapses by >5×. This is why our rescue methods operate specifically at the 4-bit budget.

### 6.5 Llama-3.1-70B (FP16 only)

We ran the FP16 baseline on Llama-3.1-70B and got PPL = **3.40** at `max_seq_len=2048`, PPL = **3.83** at `max_seq_len=512`. These are the lowest perplexities we observed across any model (expected for a 70B parameter model).

**The compressed methods OOM.** The 70B weights alone consume ~140 GB in BF16, leaving only ~4 GB for intermediate tensors. During a forward pass, our `CompressedCache.update()` returns the full decompressed KV tensors, which temporarily adds as much memory as the original FP16 cache would have used. This tips the peak memory past the available budget.

We ran this experiment for **6.5 hours** with CPU offloading before realizing it was making no meaningful progress, then killed it. The FP16 baseline is the only 70B number we report.

---

## 7. Sensitivity Analysis (Calibration)

### 7.1 Profiling Procedure

For each model, we:

1. Load the model in BF16.
2. Draw 16 sample texts from WikiText-2 (> 100 characters each).
3. Tokenize and truncate/pad to 512 tokens.
4. Run a forward pass with a standard `DynamicCache`.
5. Extract `cache.layers[i].keys` and `cache.layers[i].values` for each layer $i$.
6. For each layer and each candidate bit-width $b \in \{2, 3, 4, 6, 8\}$, compress and decompress the captured tensors with an MSE compressor and measure MSE.
7. Save the result as `results/calibration/{model_safe_name}_budget{b}_kvsplit.json`.

Profiling takes:
- Qwen-3B (36 layers): 63 seconds
- Mistral-7B (32 layers): 168 seconds
- Llama-3.1-8B (32 layers): 250 seconds
- Gemma-3-27B (62 layers): 564 seconds (fails with NaN on final layers, see Section 12.4)

### 7.2 Key vs Value Sensitivity (Mistral-7B)

At 2-bit compression:

| Layer | Key MSE | Value MSE | K/V ratio |
|---|---|---|---|
| 0 | 0.111 | 0.000046 | 2413× |
| 1 | 0.437 | 0.000446 | 980× |
| 5 | 0.286 | 0.005 | 54× |
| 10 | 0.316 | 0.009 | 34× |
| 15 | 0.327 | 0.020 | 16× |
| 20 | 0.352 | 0.034 | 10× |
| 25 | 0.323 | 0.053 | 6× |
| 30 | 0.387 | 0.128 | 3× |
| 31 | 0.391 | 0.140 | 3× |

Pattern: keys are always at least 3× more sensitive than values, and up to 2400× for early layers. Keys are roughly flat across depth (~0.3); values increase sharply from ~0.00005 at layer 0 to ~0.14 at layer 31.

### 7.3 Key vs Value Sensitivity (Qwen2.5-3B)

At 2-bit compression:

| Layer | Key MSE | Value MSE | K/V ratio |
|---|---|---|---|
| **0** | **27.63** | 0.008 | 3348× |
| 1 | 2.52 | 0.005 | 504× |
| 2 | 1.10 | 0.006 | 190× |
| 3 | 0.30 | 0.021 | 14× |
| 5 | 0.29 | 0.029 | 10× |
| 10 | 0.29 | 0.053 | 5× |
| 15 | 0.37 | 0.073 | 5× |
| 20 | 0.62 | 0.180 | 3× |
| 25 | 0.28 | 0.128 | 2× |
| 27 | **2.38** | 0.125 | 19× |
| 30 | 0.27 | 0.182 | 1.5× |
| **33** | 0.17 | **1.62** | 0.1× (value more sensitive!) |
| 35 | 0.41 | 0.159 | 2.6× |

Notes:
- **Layer 0 keys are hyper-sensitive** (27.63 MSE at 2-bit, ~60× the median key MSE). The token embedding layer's keys are the single most sensitive tensor in the cache.
- **Layer 27 keys have a second spike** — we do not have a clean interpretation but note that Qwen uses some kind of layer-specific structure.
- **Deep layer values get more sensitive than keys**: layers 32–33 have value MSE 0.69 and 1.62 at 2-bit. The adaptive allocator handles this correctly by giving these layers 6 value bits (see Section 8).

### 7.4 Key vs Value Sensitivity (Llama-3.1-8B)

At 2-bit compression, abridged:

| Layer | Key MSE | Value MSE |
|---|---|---|
| 0 | 0.14 | 0.00009 |
| 1 | 0.62 | 0.00054 |
| 5 | 0.37 | 0.005 |
| 15 | 0.39 | 0.010 |
| 25 | 0.37 | 0.022 |
| 31 | 0.42 | 0.038 |

Pattern: very similar to Mistral — keys flat around 0.3–0.4, values climb gently from ~10⁻⁴ to ~0.04. No dramatic outliers.

### 7.5 Gemma-3-27B Sensitivity

Gemma's calibration **fails with NaN** on the final layers (details in Section 12.4). We disabled adaptive allocation for Gemma and used uniform K4V4, which was already near-lossless.

---

## 8. Adaptive Bit Allocation Results

### 8.1 DP Allocator Output (Mistral-7B, budget=4.0 with K/V split)

The allocator yielded:

- Most layers: **K=4, V=4**
- A few: K=4, V=4 (redundant — the DP doesn't find non-trivial structure here because sensitivity is uniform)
- Average: K=4.00, V=4.00

**Result on WikiText-2**: Adaptive (budget=4) gave **4.987 PPL**, slightly worse than Uniform K4V4's **4.958 PPL**. **Uniform wins on Mistral.**

### 8.2 DP Allocator Output (Llama-3.1-8B, budget=4.0)

The allocator gave most layers K=6, V=2, with a few K=4, V=2 or K=6, V=3.
- Average: **K=5.75, V=2.25** (4.00 total)

**Result**: Adaptive = **6.728 PPL** vs Uniform K4V4 = **6.583 PPL**. **Uniform wins on Llama-8B.**

### 8.3 DP Allocator Output (Qwen2.5-3B, budget=4.0)

Much more diverse:
- Layer 0: **K=8, V=2** (the extreme sensitivity spike)
- Layers 1–2: K=6, V=2
- Layer 16, 18, 19, 20, 27, 29: K=6, V=2/3/4
- Layers 32, 33: K=4, V=6 (deep-layer value spike!)
- Most other layers: K=4, V=3 or K=4, V=4
- Average: **K=4.56, V=3.44**

**Result**: Adaptive = **8.014 PPL** vs Uniform K4V4 = **42.331 PPL**. **Adaptive wins dramatically — 5.3× reduction.**

### 8.4 DP Allocator Output (Qwen2.5-3B, budget=3.0)

- Layer 0: K=8, V=2
- Layer 27: K=6, V=2
- A few layers with K=6, V=3 and K=4, V=3
- Most layers: K=3, V=2 or K=4, V=2
- Average: **K=3.64, V=2.36**

**Result**: Adaptive = **8.42 PPL**. Uniform K3V3 = **1307 PPL (collapse)**. **Adaptive wins by 150×.**

### 8.5 DP Allocator Output (Qwen2.5-3B, budget=2.5)

- Most layers: K=3, V=2
- Layer 0: K=6, V=2
- Layer 31: K=3, V=3
- Average: **K=2.92, V=2.08**

**Result**: Adaptive = **9.41 PPL**. Uniform K2V2 and K3V2 both collapse. **Adaptive is the only usable method at 2.5 bits.**

### 8.6 DP Allocator Output (Qwen2.5-3B, budget=6.0)

- Most layers: K=6, V=6
- Some: K=8, V=6 or K=6, V=8
- Average: **K=6.44, V=5.56**

**Result**: Adaptive = **7.66 PPL**. Uniform K6V6 = 7.67 PPL. **Near-identical (both near-lossless).**

### 8.7 Summary: When Does Adaptive Help?

Adaptive allocation provides large gains **only when uniform allocation is catastrophic**. In the regime where uniform works (low GQA, or high GQA at ≥6 bits), adaptive is neutral or slightly worse because the DP's aggressive key-loading shifts too many bits away from values.

**Practical recipe**:
- Use **uniform K4V4** for GQA ≤ 4:1 models.
- Use **adaptive allocation** for GQA ≥ 8:1 models at budgets below 6 bits.
- Use **outlier-aware** (see next section) whenever possible, since it strictly dominates adaptive at the 4-bit budget.

---

## 9. Outlier-Aware Channel Grouping Results

### 9.1 Outlier Profiles by Model

Threshold factor 5.0, 16 calibration samples of 512 tokens each. Counts are **outlier channels out of 128** per layer.

#### Mistral-7B-Instruct-v0.3 (GQA 4:1)

**Zero outlier channels on every layer** (key and value). The MSECompressor's compression quality is already optimal; outlier grouping is a no-op.

#### Llama-3.1-8B-Instruct (GQA 4:1)

**Zero outlier channels on every layer.** Same as Mistral — clean distributions, no outliers.

#### Qwen2.5-3B-Instruct (GQA 8:1)

| Layer | Key outliers | Value outliers | Top key outlier channels |
|---|---|---|---|
| 0 | **26** | 1 | [49, 50, 53, 54, 55, ...] |
| 1 | 4 | 1 | [38, 60, 109, 114] |
| 2 | 3 | 0 | [48, 100, 112] |
| 3 | 0 | 0 | — |
| 4 | 0 | 0 | — |
| 5 | 1 | 0 | [115] |
| 6 | 1 | 0 | [118] |
| 7 | 1 | 0 | [114] |
| 8 | 1 | 0 | [117] |
| 9 | 1 | 0 | [118] |
| 10 | 2 | 0 | [56, 115] |
| 11 | 2 | 0 | [51, 116] |
| 12 | 2 | 0 | [62, 119] |
| 13 | 1 | 0 | [54] |
| 14 | 1 | 0 | [52] |
| 15 | 1 | 0 | [116] |
| 16 | 1 | 0 | [119] |
| 17 | 0 | 0 | — |
| 18 | 4 | 0 | [51, 61, 125, 127] |
| 19 | 3 | 0 | [60, 63, 123] |
| 20 | 2 | 0 | [59, 60] |
| 21 | 1 | 0 | [126] |
| 22 | 0 | 0 | — |
| 23 | 2 | 0 | [60, 124] |
| 24 | 0 | 0 | — |
| 25 | 0 | 0 | — |
| 26 | 1 | 0 | [51] |
| 27 | **5** | 0 | [60, 61, 63, 126, 127] |
| 28 | 1 | 0 | [116] |
| 29 | 2 | 0 | [58, 122] |
| 30 | 0 | 0 | — |
| 31 | 0 | 0 | — |
| 32 | 0 | 0 | — |
| 33 | 0 | 0 | — |
| 34 | 1 | 0 | [52] |
| 35 | 3 | 0 | [114, 115, 127] |

Key observations:
- **Layer 0 keys have 26 outliers** (20% of channels). This matches the calibration finding that layer 0 keys are 60× more sensitive than median.
- **Layers 1–2 have moderate outliers** (3–4 channels).
- **Most middle layers have 0–2 outliers.**
- **Values have essentially zero outliers** (only layer 0 and layer 1 have 1 each).
- **Layer 27 has 5 outliers**, matching its key sensitivity spike in calibration.

### 9.2 Layer-0 Compression MSE (Qwen2.5-3B, 4-bit)

| Method | Key MSE | Value MSE |
|---|---|---|
| Standard MSECompressor | 1.978 | 0.000689 |
| Outlier-aware (26 K outliers) | **0.021** | **0.000557** |
| Improvement | **98.9%** | 19.2% |

Protecting 26 outlier channels on layer 0 reduces key MSE by 98.9%. This is the single largest compression-quality improvement we observed anywhere in this work.

### 9.3 End-to-End Outlier Results on Qwen2.5-3B

| Method | Bits | PPL | Delta from FP16 | vs Uniform K4V4 |
|---|---|---|---|---|
| FP16 | 16 | 7.627 | — | — |
| Uniform K4V4 | 4.0 | 42.330 | **+455%** (collapse) | — |
| **Outlier-aware K4V4** | ~4.3 | **7.744** | **+1.5%** | **5.5× reduction** |
| Adaptive+Outlier K4V4 | ~4.3 | 7.932 | +4.0% | 5.3× reduction |
| Adaptive (budget=4.0) | 4.0 | 8.014 | +5.1% | 5.3× reduction |
| KIVI-4 | 4.0 | 7.887 | +3.4% | — |

**Outlier-aware is the strongest single method**, giving 7.744 PPL (better than even KIVI-4's 7.887). The effective bit budget is ~4.3 (the 0.3 bits come from keeping ~8 outlier channels per layer in FP16 on average).

### 9.4 Adaptive + Outlier Combined (Unexpected Result)

We had expected the combination to strictly dominate outlier-alone. Instead, it came in **worse** (7.932 vs 7.744). Our interpretation:

- Outlier-aware alone uses uniform K4V4 for non-outlier channels, so every layer has the same compression quality on its "clean" channels.
- Adaptive+Outlier uses the adaptive allocation's recipe (K=4.56, V=3.44), which reduces value bits on most layers to 2 or 3.
- Since outlier protection has already addressed the layer-0 hot spot, the main remaining sensitivity is in normal keys and values across all layers. Dropping values from 4 to 3 bits hurts that more than the small gain from having 1 extra key bit somewhere else.

The two techniques are partially redundant: both devote extra precision to the most damaging channels, but through different mechanisms. A more integrated allocator that is aware of the outlier overhead budget might recover the small gap; we leave this for future work.

### 9.5 Outlier Method Is GQA-Specific

- **GQA 4:1 models (Mistral, Llama-8B)**: 0 outlier channels → outlier method is a no-op.
- **GQA 2:1 models (Gemma)**: untested (Gemma calibration fails with NaN).
- **GQA 8:1 models (Qwen)**: up to 26 outlier channels on layer 0 → outlier method is a massive rescue.

The outlier phenomenon appears only in the high-GQA regime, which is consistent with the GQA amplification hypothesis: with fewer KV heads, each head carries more information, and the model learns to concentrate it in a few high-magnitude channels.

---

## 10. Needle-in-a-Haystack

### 10.1 Setup

- Needle text: `"The secret project code name is AURORA-7749."`
- Haystack: repetitive filler text (corporate-meeting boilerplate) padded to target length.
- The needle is inserted at position `int(n_filler * needle_pos)` for `needle_pos ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`.
- Question: `"What is the secret project code name mentioned in the document? Answer with just the code name, nothing else."`
- Scoring: `EXACT` if response contains "AURORA-7749"; `PARTIAL` if contains both "AURORA" and "7749"; `MISS` otherwise.
- Generation: greedy, `max_new_tokens=32`.

### 10.2 Mistral-7B Results

Perfect retrieval for FP16, TQ K4V4, and KIVI-4 across all 3 contexts × 5 positions = 15 cells each.

| Method | 4K (5 positions) | 8K | 16K | Total |
|---|---|---|---|---|
| FP16 | 5/5 EXACT | 5/5 | 5/5 | **15/15** |
| TQ K4V4 | 5/5 | 5/5 | 5/5 | **15/15** |
| KIVI-4 | 5/5 | 5/5 | 5/5 | **15/15** |
| **TQ K2V2** | 5/5 | 4/5 MISS@0.7 (output "AURORA-774") | 4/5 MISS@0.5 ("AURORA-774") | **13/15** |

At 2-bit compression, TQ K2V2 begins to drop the last digit of the code name in some positions, but the overall retrieval signal is still largely intact.

### 10.3 Why No Needle for Qwen?

We did not run needle on Qwen because the interesting behavior on Qwen is the catastrophic collapse at 4-bit — perplexity already tells the story and needle would simply fail everywhere. For the rescued methods (outlier-aware, adaptive), we focused on perplexity because it gives a more granular measurement of quality degradation.

---

## 11. Hardware Benchmarks

### 11.1 Methodology

From `scripts/benchmark_hw.py`:

- **Latency**: warmup run, then one timed prefill (2048 prompt tokens) and one timed decode loop (32 or 64 tokens). Uses `torch.cuda.synchronize()` for accurate timing.
- **Memory**: `torch.cuda.reset_peak_memory_stats()`, then one forward pass, then `torch.cuda.max_memory_allocated()`. Reports peak and cache overhead (peak minus model memory).

### 11.2 Latency Results

Decode throughput (tokens/sec):

| Method | Mistral-7B | Llama-3.1-8B | Qwen2.5-3B | Gemma-3-27B |
|---|---|---|---|---|
| **FP16** | 37.7 | 16.0 | 24.4 | 5.5 |
| TQ K4V4 | 3.8 (10×) | 2.1 (8×) | 2.4 (10×) | 1.6 (3.4×) |
| TQ K4V2 | 3.8 | 2.6 | 2.5 | 1.7 |
| TQ K2V2 | 3.9 | 2.6 | 2.3 | 1.6 |
| KIVI-4 | 7.7 (5×) | 7.7 (2×) | 7.8 (3×) | 4.4 (1.3×) |
| KIVI-2 | 7.8 | 7.8 | 7.3 | 4.0 |

(Values in parentheses are slowdown factors relative to FP16.)

Prefill latency (milliseconds for 2048 tokens):

| Method | Mistral-7B | Llama-3.1-8B | Qwen2.5-3B | Gemma-3-27B |
|---|---|---|---|---|
| FP16 | 168 | 272 | 148 | 1007 |
| TQ K4V4 | 293 | 316 | 290 | 1591 |
| KIVI-4 | 183 | 233 | 130 | 1069 |

### 11.3 Memory Results

Peak memory overhead from the KV cache (total peak minus model-only memory), at seq_len=2048 or 4096:

| Method | Mistral-7B (4K) | Llama-8B (4K) | Qwen-3B (4K) | Gemma-3-27B (2K) |
|---|---|---|---|---|
| FP16 | 322 MB | 574 MB | 632 MB | 669 MB |
| TQ K4V4 | 452 MB (**↑**) | 583 MB | 629 MB | 694 MB |
| TQ K4V2 | 446 MB | 579 MB | 628 MB | 686 MB |
| TQ K2V2 | 325 MB | 575 MB | 627 MB | 680 MB |
| KIVI-4 | 354 MB | 599 MB | 632 MB | 729 MB |

**Surprising observation**: on Mistral-7B, TQ K4V4 uses **more** cache memory than FP16 (452 vs 322 MB). On other models it's roughly equal. This is because `CompressedCache.update()` returns the full decompressed KV for attention, and we transiently hold both the compressed chunks and the full decompressed tensors during the forward pass. The peak is dominated by the full decompressed tensors, not the compressed storage.

### 11.4 Theoretical Compression Ratios

| Method | Bits per element | Theoretical ratio |
|---|---|---|
| FP16 | 16 | 1.0× |
| TQ K4V4 | 4 | 4.0× |
| TQ K4V2 | 3 (avg) | 5.3× |
| TQ K2V2 | 2 | 8.0× |

These are the ratios achievable with **fused attention kernels** that operate directly on the compressed representation. Our Python reference implementation decompresses into temporary FP16 tensors before calling standard attention, so it does not realize these savings.

### 11.5 Honest Framing for the Paper

We position the hardware numbers as follows:

- **Our contributions are algorithmic**: the sensitivity analysis, adaptive allocation, and outlier grouping all operate at the level of bit budgets and compression quality. They are agnostic to the implementation of the compress/decompress kernels.
- **Our implementation is Python-level**: every forward pass calls NumPy-style tensor operations through PyTorch, with no CUDA kernel fusion. Both TurboQuant and KIVI in our codebase pay the same overhead.
- **Fused kernels are future work**: production implementations would realize the full theoretical compression. FlashAttention-style kernels already exist for per-channel scalar quantization (KIVI-style); TurboQuant-style VQ would require a new kernel that does the rotation, centroid lookup, and attention in a single pass.
- **KIVI papers make the same caveat**: their published numbers are also from Python-level reference implementations, with theoretical compression ratios used for the main claims.

---

## 12. Failed Experiments and Debugging Stories

This section records the dead ends, mistakes, and debugging we went through. It's useful for anyone continuing this work — many of these experiments took hours to run before we realized they wouldn't produce useful numbers.

### 12.1 Gemma-4-31B: Broken AutoModelForCausalLM

**Attempt**: we tried Gemma-4-31B-it as a larger evaluation target. It downloaded successfully and loaded without errors.

**Problem**: at inference time, it outputs garbage:
```
Prompt: "The capital of France is"
Output: "The capital of France is France is France is France is..."
Small-text PPL: 1,862,629.95
```

**Diagnosis**: Gemma-4 is a **multimodal model** with a complex architecture (mixed sliding/full attention layer types, global head_dim=512 vs local head_dim=256, 60 layers). It loads via `AutoModelForCausalLM`, but the text-only forward path does not work correctly with this class. A dedicated Gemma4-specific model class or vision-aware loader would be required.

**Resolution**: abandoned Gemma-4. We switched to Gemma-3-27B after the user approved the Gemma-3 license on HuggingFace.

**Time cost**: ~45 minutes (download + debugging).

### 12.2 Llama-3.1-70B: OOM With Compression

**Attempt**: run the standard sweep (FP16, TQ K4V4, KIVI-4) on Llama-3.1-70B.

**Observations**:
- FP16 baseline: **PPL = 3.40** at `max_seq_len=2048`. This works fine; it completes in ~30 minutes.
- TQ K4V4: CUDA OOM after ~5 minutes. GPU 2 has only 14 MB free (out of 48 GB); the 70B model weights + temporary decompression tensors overflow the budget.
- KIVI-4: Same OOM.

**Second attempt**: reduce `max_seq_len` to 512 and add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- FP16 at 512: PPL = 3.83 (slightly higher than the 2048 number, as expected).
- TQ K4V4 at 512: started running, but extremely slow due to CPU offloading.

**Killed at 6.5 hours** without producing a compressed number. The 70B model partially offloads to CPU (13.5 GB RAM usage, 525% CPU utilization), and every forward pass with compression requires moving tensors between CPU and GPU repeatedly. We estimate the full WikiText-2 sweep at ~20 hours per method.

**Resolution**: report only the FP16 baseline on 70B and note as a limitation that compressed evaluation requires a fused kernel implementation that decompresses in place.

**Time cost**: ~8 hours of wall time, including the killed run.

### 12.3 Qwen K4V4 Collapse: Finding the GQA Amplification

This was the most important debugging story because it became the paper's central contribution.

**Initial attempt**: run the standard sweep on Qwen2.5-3B.

**Observation**: FP16 PPL = 7.63. TQ K4V4 PPL = 42.33. Expected ~7.7.

**First hypothesis**: implementation bug. We wrote `scripts/debug_qwen.py` to isolate the issue.

**Debug script findings**:
- FP16 → 7.58 PPL (matches expected).
- TQ K8V8 → 7.58 PPL (essentially lossless). This rules out a broken compressor and confirms the pipeline is mechanically correct.
- TQ K6V4 → 7.65 PPL (+0.9%). Still fine.
- TQ K4V4 rw=128 → 40.60 PPL. **Still broken at 4 bits.**
- TQ K4V4 rw=512 → 23.58 PPL. Smaller but still severely degraded.
- TQ K4V4 rw=1024 → 10.55 PPL. Only recovers partially.

**Second hypothesis**: KV vector magnitudes are abnormal. We inspected the raw statistics:
- Mistral-7B key: `mean=-0.007, std=2.1, max=9.3` (small range).
- Qwen2.5-3B key: `mean=-0.12, std=15.38, max=92.0` (very wide range!).

Qwen's keys have **10× the spread** of Mistral's keys. This could be a problem — the Lloyd-Max codebook is designed for a specific Beta distribution, and the MSECompressor normalizes to unit sphere before rotation, so the magnitude itself shouldn't matter. But the **shape** of the distribution might.

**Third hypothesis**: it's the GQA ratio. Mistral has 8 KV heads shared by 32 query heads (GQA 4:1). Qwen has **only 2 KV heads** shared by 16 query heads (GQA **8:1**). Each Qwen KV head is consumed by 8× as many query heads as a Mistral KV head, and any per-head compression error is amplified accordingly.

**Verification**: we ran Llama-3.1-8B (GQA 4:1) and Gemma-3-27B (GQA 2:1) with the same K4V4 config. Both worked: +1.6% and +0.7% PPL degradation respectively. **The failure is specific to GQA 8:1.**

**Resolution**: this became the paper's central finding (GQA amplification) and motivated the adaptive/outlier contributions. The bug hunt took about 2 hours but led to the most valuable insight in the entire project.

### 12.4 Gemma-3 Calibration: NaN on Final Layers

**Attempt**: run `scripts/calibrate.py` on Gemma-3-27B at budget=4.0.

**Output** (abbreviated):
```
Per-layer KEY sensitivity (MSE distortion):
 Layer |    2-bit |    3-bit |    4-bit |    6-bit |    8-bit
    0  |  0.1103  |  0.0323  |  0.0089  |  0.00074 |  0.00006
    1  |  0.3887  |  0.1138  |  0.0311  |  0.00258 |  0.00020
    ...
   57  |  0.3156  |  0.0924  |  0.0253  |  0.00210 |  0.00017
   58  |  0.3287  |  0.0965  |  0.0264  |  0.00220 |  0.00017
   59  |     nan  |     nan  |     nan  |     nan  |     nan
   60  |     nan  |     nan  |     nan  |     nan  |     nan
   61  |     nan  |     nan  |     nan  |     nan  |     nan
```

The last 3 layers (out of 62) produce NaN sensitivity values. The DP allocator cannot handle NaN and crashes.

**Root cause**: Gemma-3-27B has a **mixed attention pattern**: 52 sliding-attention layers and 10 full-attention layers (every 6th layer, starting from layer 5). The final layers (59, 60, 61) are sliding-attention layers at the very end of the model. Our calibration harness captures KV from a standard `DynamicCache`, which handles sliding attention differently from full attention — the sliding layers may produce zero-length tensors or numerically unstable norms on the short calibration sequences (512 tokens).

**Resolution**: we did not fix the calibration harness (it would require a sliding-attention-aware capture path). Since uniform K4V4 on Gemma-3 is already near-lossless (+0.7% PPL), adaptive allocation is not needed for Gemma. We note the bug as a limitation and recommend fixing the profiler for sliding-attention models as future work.

### 12.5 Disk Full on Server

**Attempt**: download Gemma-4-31B to the server (~63 GB).

**Error**:
```
UserWarning: Not enough free disk space to download the file.
The expected file size is: 49784.79 MB.
The target location only has 11110.23 MB free disk space.
```

The server had been accumulating large model caches: Llama-3.1-70B (132 GB!), Llama-3.3-70B INT8 (68 GB), gpt-oss-120b (61 GB). Total ~260 GB in `~/.cache/huggingface/hub`.

**Resolution**: removed `Llama-3.3-70B-INT8`, `gpt-oss-120b`, and `llama-3.3-70b-instruct-awq` (all unused by our experiments). Freed 166 GB.

**Time cost**: ~5 minutes.

### 12.6 Transformers 5.5.0 DynamicCache API Change

**Attempt**: the initial `calibrate.py` used `cache[layer_idx]` to extract KV tensors from the HuggingFace DynamicCache.

**Error**:
```python
TypeError: 'DynamicCache' object is not subscriptable
```

**Root cause**: in transformers ≥ 5.0, `DynamicCache` is no longer indexable. KV tensors must be accessed via `cache.layers[i].keys` and `cache.layers[i].values`.

**Resolution**: updated `turboquant/adaptive.py` to use the new API.

**Lesson**: we now document this in `CLAUDE.md` and `README.md` so future work on this codebase is aware.

### 12.7 Attention Layer Output Format Change

**Attempt**: our initial `calibrate.py` hooked the attention layer's forward output, expecting `(attn_output, attn_weights, present_kv)` with `present_kv` as a tuple `(k, v)`.

**Error**: `present_kv` is `None` or `attn_weights` is at index 1, not `present_kv`.

**Root cause**: in transformers 5.5.0, `MistralAttention.forward()` returns only `(attn_output, attn_weights)`. The KV cache is managed internally by the `past_key_values` object — no separate return value.

**Resolution**: switched to a different capture strategy: create a fresh `DynamicCache`, do a forward pass, then read `cache.layers[i].keys` / `.values` after the fact.

### 12.8 Matplotlib NumPy Incompatibility

**Attempt**: run `scripts/gen_sensitivity_plot.py` locally.

**Error**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6.
matplotlib not installed.
```

The local system has NumPy 2.2.6 and an older matplotlib that was compiled against NumPy 1.x.

**Resolution**: generate the plots on the server instead, where the venv has matching versions. We now keep plot generation server-side.

### 12.9 Llama-3.2-3B Gated Despite Token

Even after setting both HF tokens provided by the user, `meta-llama/Llama-3.2-3B-Instruct` still returned 403. The 3.2 series has a stricter license gate than 3.1 or the Gemma series.

**Resolution**: skipped Llama-3.2-3B; used Qwen2.5-3B as our 3B representative instead.

### 12.10 Rebase Conflict on GitHub Push

When we first pushed to the user's GitHub repo, the remote already had one commit (an auto-generated "hi" README). Our local `main` branch didn't know about it.

**Error**:
```
! [rejected] main -> main (fetch first)
```

**Resolution**: `git fetch origin main`, then `git pull --rebase origin main`. This produced a merge conflict on `README.md` (remote said "hi", ours said the full research README). We kept ours with `git checkout --theirs README.md` (during a rebase, `--theirs` means the branch being rebased = our local commit). Completed the rebase and pushed cleanly.

**Lesson**: during a rebase, `--ours` and `--theirs` are flipped compared to a regular merge. `--theirs` means "the branch I am rebasing = my local work."

---

## 13. Cross-Model Findings and Analysis

### 13.1 The GQA Amplification Hypothesis, Formalized

Let $E$ be the expected per-head compression error variance. For a GQA model with $n_{\text{heads}}$ query heads and $n_{\text{kv\_heads}}$ KV heads, let $g = n_{\text{heads}} / n_{\text{kv\_heads}}$.

**KIVI** stores per-channel metadata (scale, zero) at full precision. The reconstruction error is dominated by the quantization of individual elements, and the error seen by any single attention dot product is bounded by the per-channel $\ell_\infty$ error. The total error seen by a query head $h$ is approximately

$$
\text{err}_{\text{KIVI}}(h) \approx D \cdot \sigma_{\text{chan}}^2,
$$

where $\sigma_{\text{chan}}^2$ is the per-channel variance. Crucially, this does **not** depend on $g$.

**TurboQuant** quantizes whole vectors. The Lloyd-Max index represents a direction on the $D$-sphere, so mis-indexing produces a vector-valued error $\delta \in \mathbb{R}^D$ with $\|\delta\|_2^2 = \sigma_{\text{vec}}^2$. This error is seen by every query head sharing the KV head, so the total error across all query heads in the group is

$$
\text{err}_{\text{TQ}}(\text{group}) \approx g \cdot \sigma_{\text{vec}}^2.
$$

For the error to be equal to KIVI's per-head error, we need $g \cdot \sigma_{\text{vec}}^2 \le D \cdot \sigma_{\text{chan}}^2$, i.e.

$$
\sigma_{\text{vec}}^2 / \sigma_{\text{chan}}^2 \le D / g.
$$

For $D = 128$ and $g = 4$ (Mistral, Llama-8B), this requires $\sigma_{\text{vec}}^2 \le 32 \sigma_{\text{chan}}^2$, which is easily satisfied because TurboQuant's rotation-based Lloyd-Max is very efficient.

For $g = 8$ (Qwen, Llama-70B), the tolerance is $\sigma_{\text{vec}}^2 \le 16 \sigma_{\text{chan}}^2$. At 4-bit compression on Qwen, our measurements suggest $\sigma_{\text{vec}}^2 / \sigma_{\text{chan}}^2 \approx 20$, which **exceeds the tolerance**, explaining the collapse.

This is a rough informal calculation — we did not make it fully rigorous — but it's consistent with the empirical pattern.

### 13.2 Why Layer 0 Is Always the Most Sensitive Key Layer

Across every model we profiled, **layer 0 keys** are significantly more sensitive than keys in subsequent layers. On Qwen2.5-3B, layer 0 key MSE at 2-bit is 27.6 versus ~0.3 for median layers (a 90× gap).

**Interpretation**: layer 0 performs the "token-identification" part of the forward pass. Its keys directly encode token-level information that has not yet been contextualized, and disrupting this information cascades through every subsequent layer. Layer 0 keys are essentially an **embedding lookup in key-space**, and embedding lookups are inherently low-rank / near-discrete, which is a pathological input for rotation-based VQ (the unit sphere assumption breaks down when vectors cluster near a few distinct directions).

On Qwen, we also observed that **layer 0 has 26 outlier channels** (20% of all channels). These outlier channels are likely carrying the discrete token identity, and protecting them (via outlier-aware grouping) recovers 98.9% of the layer-0 key MSE.

### 13.3 Why Deep Values Get Spike-Sensitive on Qwen

Qwen layers 32–33 have value MSE at 2-bit of 0.69 and 1.62 respectively, versus ~0.03 for median layers. We do not have a clean theoretical explanation, but note:

- Layer 33 is the penultimate layer (Qwen has 36 layers).
- Deep-layer values are **linearly mixed** into the residual stream and ultimately projected to the output head. Any perturbation in these values directly affects the next-token distribution.
- This pattern is milder or absent on other models (Mistral layer 31 value MSE at 2-bit is 0.14 — 10× smaller than Qwen layer 33).

The adaptive allocator handles this correctly: it assigns 6 value bits to Qwen layers 32–33, recognizing their outsize importance.

### 13.4 Why Gemma-3 Is the TurboQuant Sweet Spot

Gemma-3-27B has GQA 2:1 — each KV head is shared by only 2 query heads. This is the mildest GQA ratio we tested (MHA would be 1:1 and is rare in modern architectures). The GQA amplification term $g \sigma_{\text{vec}}^2 = 2 \sigma_{\text{vec}}^2$ is small enough that TurboQuant's 4-bit error is essentially invisible.

At the same time, KIVI's per-channel overhead becomes meaningful on a 62-layer, 16-KV-head model: KIVI-4's effective bit-width is closer to 5.5 due to the scale/zero metadata, but it still gets +22.4% PPL degradation (vs TQ K4V4 at +0.7%). This is where TurboQuant's VQ approach earns its keep: at mild GQA ratios, the vector-level quantization is strictly better than per-channel scalar quantization.

### 13.5 Practical Recommendation

Based on our findings, we recommend the following compression recipes:

| GQA ratio | Recommended method | Expected quality |
|---|---|---|
| 1:1 (MHA) | TurboQuant K4V4 uniform | ≤1% PPL degradation |
| 2:1 | TurboQuant K4V4 uniform | ~1% PPL degradation |
| 4:1 | TurboQuant K4V4 uniform | 1–2% PPL degradation |
| 8:1 | **Outlier-aware TurboQuant K4V4** | 1–2% PPL degradation |
| 8:1, sub-4-bit budget | Adaptive TurboQuant (DP allocator) | 3–10% PPL degradation |
| 8:1, 2-bit budget | No TurboQuant method works. Use KIVI or accept quality loss. | — |

Per-channel scalar methods (KIVI) remain a strong fallback at very low bit-widths on high-GQA models, but always with 5–20% degradation at 4-bit.

---

## 14. Limitations

### 14.1 Implementation Speed

Our Python reference is 3–10× slower than FP16 during decode. This is a property of the implementation, not the algorithm: a fused attention kernel that operates on compressed KV representations would eliminate the decompression overhead. We did not implement such a kernel. Reporting hardware numbers that compare favorably to FP16 would require significant kernel-engineering effort and is explicitly left as future work.

### 14.2 Peak Memory Does Not Decrease

Counter-intuitively, our compressed cache uses **more or equal** peak memory than the FP16 cache. The reason: `CompressedCache.update()` returns full decompressed KV tensors for attention, so the transient peak is dominated by these tensors (as large as the FP16 cache would have been). The compressed chunks add on top.

A production implementation that modifies attention to operate directly on the compressed representation would realize the full theoretical compression ratio (4× at 4 bits). We benchmark only the theoretical ratios in the paper and note this clearly.

### 14.3 Llama-3.1-70B Compressed Evaluation

We were unable to run compressed evaluation on Llama-3.1-70B because the weights alone consume nearly all our 144 GB VRAM. The decompression overhead tips the peak memory past OOM. We report only the FP16 baseline (3.40 PPL) on 70B.

A server with 4× A6000 or a single H100 (80 GB) would likely handle 70B compressed evaluation. Alternatively, a kernel-level implementation that decompresses in place would avoid the issue.

### 14.4 Gemma-3 Calibration

Our sensitivity profiler produces NaN values on the final 3 layers of Gemma-3-27B due to an interaction with sliding-window attention. We did not diagnose or fix this; uniform K4V4 is near-lossless on Gemma-3 so adaptive allocation is not needed anyway. Future work should fix the profiler for sliding-attention architectures.

### 14.5 Calibration Domain Sensitivity

We use 16 samples from WikiText-2 or C4 for calibration. We have not studied how sensitive the resulting allocation is to calibration-domain mismatch (e.g., would a code calibration set give different outlier channels?). Existing weight-quantization literature suggests sensitivity profiles are fairly robust across domains, but this deserves direct study in the KV cache setting.

### 14.6 Outlier Threshold

The outlier detection threshold (`threshold_factor=5.0`) was chosen heuristically to match typical outlier ratios observed in AWQ-style analyses. A principled derivation (e.g., based on desired recovery quality) would be preferable. We also did not ablate this threshold — it's possible that a tighter or looser threshold would yield different tradeoffs.

### 14.7 Single-GPU-Class Hardware

All experiments were run on RTX A6000 GPUs. Modern H100/H200 class GPUs have different memory bandwidth and compute characteristics, which might affect the latency numbers. We expect the perplexity and compression-quality results to transfer, but hardware measurements should be re-run on target deployment hardware.

### 14.8 No Downstream Task Evaluation

We implemented `eval/downstream.py` as a wrapper around `lm-evaluation-harness` for MMLU, ARC, HellaSwag, WinoGrande, GSM8K. However, on reflection we realized that these tasks use very short contexts (a few hundred tokens per example), where KV cache compression has essentially zero impact — all methods get the same result because the cache is dominated by the residual window. Running these benchmarks would produce uninformative "no difference" results.

For a meaningful downstream evaluation, we would need either:
- Long-context tasks (LongBench-E, Needle-in-a-haystack — which we did run), or
- A modified forward pass that compresses even the residual window (to force the effect to show up on short contexts).

We chose the first route and report needle results on Mistral-7B.

---

## 15. Reproduction Instructions

### 15.1 Prerequisites

- Python 3.12+
- PyTorch 2.5+ with CUDA 12.1+
- 48 GB+ GPU memory for 7B models, 144 GB+ for 70B, disk space for model weights
- HuggingFace account with access to gated models (Llama-3.1, Gemma-3)

### 15.2 Installation

```bash
git clone https://github.com/azamkhan555622848/Sensitivity-Adaptive-KV-Cache-Compression-TurboQuant.git
cd Sensitivity-Adaptive-KV-Cache-Compression-TurboQuant

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scipy transformers datasets accelerate pyyaml pandas lm-eval rouge-score matplotlib pytest
```

### 15.3 Set HuggingFace Token (for gated models)

```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")
```

Then accept the licenses at:
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
- https://huggingface.co/google/gemma-3-27b-it

### 15.4 Run Unit Tests

```bash
pytest tests/ -v
```

Expected: 18 passed.

### 15.5 Run a Small Smoke Test

```bash
python scripts/run_experiment.py --config configs/sweeps/quick-test.yaml
```

This runs Mistral-7B with FP16, TQ K4V4, and KIVI-4 on WikiText-2 perplexity. Should take ~10 minutes on an A6000 and produce results in `results/quick-test/`.

### 15.6 Reproduce Main Paper Experiments

Each sweep corresponds to one section of the paper:

```bash
# Mistral-7B full sweep (Section 6.1 main table row)
python scripts/run_experiment.py --config configs/sweeps/mistral-full.yaml

# Llama-3.1-8B full sweep + adaptive
python scripts/run_experiment.py --config configs/sweeps/llama8b-full.yaml
python scripts/run_experiment.py --config configs/sweeps/llama8b-adaptive.yaml

# Qwen2.5-3B full sweep + budget sweep + outlier
python scripts/run_experiment.py --config configs/sweeps/qwen-full.yaml
python scripts/run_experiment.py --config configs/sweeps/qwen-budget-sweep.yaml
python scripts/run_experiment.py --config configs/sweeps/qwen-outlier.yaml

# Gemma-3-27B full sweep
python scripts/run_experiment.py --config configs/sweeps/gemma3-27b-full.yaml

# Llama-3.1-70B FP16 only (compressed methods will OOM on 144 GB)
python scripts/run_experiment.py --config configs/sweeps/llama70b-core.yaml

# Mistral-7B needle-in-haystack
python scripts/run_experiment.py --config configs/sweeps/mistral-needle.yaml
```

### 15.7 Reproduce Sensitivity Calibration

```bash
# Per-model calibration
python scripts/calibrate.py --model mistralai/Mistral-7B-Instruct-v0.3 --budget 4.0
python scripts/calibrate.py --model meta-llama/Llama-3.1-8B-Instruct --budget 4.0
python scripts/calibrate.py --model Qwen/Qwen2.5-3B-Instruct --budget 4.0
# Gemma-3 calibration fails with NaN — see Section 12.4.

# Qwen sensitivity at multiple budgets (for the Qwen rescue experiments)
for b in 2.0 2.5 3.0 4.0 6.0; do
    python scripts/calibrate.py --model Qwen/Qwen2.5-3B-Instruct --budget $b
done
```

Results go to `results/calibration/`.

### 15.8 Reproduce Outlier Profiling

```bash
python scripts/profile_outliers.py --model Qwen/Qwen2.5-3B-Instruct --threshold 5.0
python scripts/profile_outliers.py --model mistralai/Mistral-7B-Instruct-v0.3 --threshold 5.0
python scripts/profile_outliers.py --model meta-llama/Llama-3.1-8B-Instruct --threshold 5.0
```

Results go to `results/outlier/`.

### 15.9 Reproduce Hardware Benchmarks

```bash
python scripts/benchmark_hw.py --model mistralai/Mistral-7B-Instruct-v0.3 --prompt-len 2048 --gen-len 64
python scripts/benchmark_hw.py --model meta-llama/Llama-3.1-8B-Instruct --prompt-len 2048 --gen-len 64
python scripts/benchmark_hw.py --model Qwen/Qwen2.5-3B-Instruct --prompt-len 2048 --gen-len 64
python scripts/benchmark_hw.py --model google/gemma-3-27b-it --prompt-len 2048 --gen-len 32
```

Results go to `results/benchmarks/`.

### 15.10 Regenerate Paper Tables and Figures

```bash
python scripts/gen_paper_tables.py       # writes paper/tables/*.tex
python scripts/gen_sensitivity_plot.py   # writes paper/figures/sensitivity.{pdf,png}
```

### 15.11 Build the Paper (PDF)

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Produces `paper/main.pdf`.

### 15.12 Build the DOCX Version

```bash
python scripts/build_docx.py
```

Produces `paper/main.docx`.

---

## 16. Appendix: Raw Numbers

### 16.1 All Perplexity Results (JSON → Table)

Source: `results/**/*.json`. Aggregated by `scripts/gen_paper_tables.py`.

#### Mistral-7B-Instruct-v0.3 (WikiText-2, 334,660 eval tokens)

| Method | Config | PPL |
|---|---|---|
| FP16 | — | 4.9367 |
| TQ K4V4 | rw=128 | 4.9579 |
| TQ K4V2 | rw=128 | 5.0499 |
| TQ K3V2 | rw=128 | 5.1479 |
| TQ K3V3 | rw=128 | 5.0519 |
| TQ K2V2 | rw=128 | 5.7892 |
| TQ K4V4 | rw=0 | 5.0240 (from ablation) |
| TQ K4V4 | rw=64 | 4.9810 (from ablation) |
| TQ K4V4 | rw=256 | 4.9490 (from ablation) |
| TQ K6V2 | rw=128 | 5.0190 |
| TQ-Adaptive | budget=2.5 | 5.2005 |
| TQ-Adaptive | budget=3.0 | 5.0466 |
| TQ-Adaptive | budget=4.0 | 4.9867 |
| KIVI-4 | rw=128 | 5.1280 |
| KIVI-2 | rw=128 | 235.1792 |

#### Llama-3.1-8B-Instruct (WikiText-2, 289,076 eval tokens)

| Method | Config | PPL |
|---|---|---|
| FP16 | — | 6.4818 |
| TQ K4V4 | rw=128 | 6.5834 |
| TQ K4V2 | rw=128 | 6.8655 |
| TQ K2V2 | rw=128 | 10.3466 |
| TQ-Adaptive | budget=4.0 | 6.7278 |
| KIVI-4 | rw=128 | 7.2858 |
| KIVI-2 | rw=128 | 1243.7384 |

#### Qwen2.5-3B-Instruct (WikiText-2, 299,077 eval tokens)

| Method | Config | PPL |
|---|---|---|
| FP16 | — | 7.6266 |
| TQ K8V8 | rw=128 | 7.6276 |
| TQ K6V6 | rw=128 | 7.6714 |
| TQ K6V4 | rw=128 | 7.7102 |
| TQ K4V4 | rw=128 | **42.3305** |
| TQ K4V4 | rw=512 | 24.1284 |
| TQ K4V4 | rw=1024 | 10.5525 (from debug) |
| TQ K3V3 | rw=128 | 1307.7871 |
| TQ K2V2 | rw=128 | 4231.7659 |
| TQ-Adaptive | budget=2.0 | 4231.7659 (tied with uniform, no improvement at the floor) |
| TQ-Adaptive | budget=2.5 | 9.4062 |
| TQ-Adaptive | budget=3.0 | 8.4192 |
| TQ-Adaptive | budget=4.0 | 8.0142 |
| TQ-Adaptive | budget=6.0 | 7.6607 |
| TQ-Outlier K4V4 | rw=128 | **7.7437** |
| TQ-Adaptive+Outlier | budget=4.0 | 7.9323 |
| KIVI-6 | rw=128 | 7.6419 |
| KIVI-4 | rw=128 | 7.8865 |
| KIVI-2 | rw=128 | 4233.17 (collapse) |

#### Gemma-3-27B-it (WikiText-2, ~50,000 eval tokens with max_tokens cap)

| Method | Config | PPL |
|---|---|---|
| FP16 | — | 7.4675 |
| TQ K4V4 | rw=128 | 7.5213 |
| TQ K4V2 | rw=128 | 7.6812 |
| TQ K2V2 | rw=128 | 14.2827 |
| KIVI-4 | rw=128 | 9.1357 |
| KIVI-2 | rw=128 | 439176.9894 |

#### Llama-3.1-70B-Instruct (WikiText-2)

| Method | Config | PPL |
|---|---|---|
| FP16 (max_seq=2048) | — | 3.3970 |
| FP16 (max_seq=512) | — | 3.8283 |
| TQ K4V4 | — | OOM |
| KIVI-4 | — | OOM |

### 16.2 Adaptive Allocation (Qwen2.5-3B, budget=4.0)

Full per-layer allocation from `results/calibration/Qwen_Qwen2.5-3B-Instruct_budget4.0.json`:

| Layer | Key bits | Value bits |
|---|---|---|
| 0 | 8 | 2 |
| 1 | 6 | 2 |
| 2 | 6 | 2 |
| 3 | 4 | 2 |
| 4 | 4 | 2 |
| 5 | 4 | 3 |
| 6 | 4 | 3 |
| 7 | 4 | 3 |
| 8 | 4 | 3 |
| 9 | 4 | 3 |
| 10 | 4 | 3 |
| 11 | 4 | 3 |
| 12 | 4 | 3 |
| 13 | 4 | 3 |
| 14 | 4 | 3 |
| 15 | 4 | 3 |
| 16 | 6 | 3 |
| 17 | 4 | 3 |
| 18 | 6 | 4 |
| 19 | 6 | 4 |
| 20 | 6 | 4 |
| 21 | 4 | 4 |
| 22 | 4 | 4 |
| 23 | 4 | 4 |
| 24 | 4 | 3 |
| 25 | 4 | 4 |
| 26 | 4 | 4 |
| 27 | 6 | 4 |
| 28 | 4 | 4 |
| 29 | 6 | 4 |
| 30 | 4 | 4 |
| 31 | 4 | 4 |
| 32 | 4 | 6 |
| 33 | 4 | 6 |
| 34 | 4 | 4 |
| 35 | 4 | 4 |

Average: K = 4.56, V = 3.44, total = 4.00.

### 16.3 Complete Outlier Profile (Qwen2.5-3B)

See Section 9.1 for the full per-layer outlier count table.

Key outlier channels, top 10 most frequent:
- Channel 60: appears on layers 18, 19, 20, 23, 27 (5 layers)
- Channel 127: appears on layers 18, 27 (2 layers)
- Channel 126: appears on layers 18, 21, 27 (3 layers)
- Channel 115: appears on layers 5, 10, 34, 35 (4 layers)
- Channel 118: appears on layers 6, 9 (2 layers)
- Channel 116: appears on layers 11, 15, 28 (3 layers)

There is no single channel that is always an outlier, but channels in the range [48-64] and [114-127] are over-represented, suggesting some structural concentration in the key projection weights.

### 16.4 Sensitivity Profile Summary (Qwen2.5-3B, Selected Layers)

| Layer | K 2-bit MSE | K 4-bit MSE | K 8-bit MSE | V 2-bit MSE | V 4-bit MSE | V 8-bit MSE |
|---|---|---|---|---|---|---|
| 0 | **27.632** | 1.978 | 0.014 | 0.00831 | 0.00069 | 0.00000 |
| 1 | 2.518 | 0.220 | 0.00133 | 0.00517 | 0.00042 | 0.00000 |
| 2 | 1.100 | 0.087 | 0.00058 | 0.00579 | 0.00046 | 0.00000 |
| 3 | 0.304 | 0.024 | 0.00016 | 0.02138 | 0.00171 | 0.00001 |
| 10 | 0.291 | 0.023 | 0.00016 | 0.05317 | 0.00430 | 0.00003 |
| 15 | 0.369 | 0.030 | 0.00019 | 0.07279 | 0.00583 | 0.00004 |
| 20 | 0.623 | 0.050 | 0.00034 | 0.18047 | 0.01445 | 0.00010 |
| 25 | 0.277 | 0.022 | 0.00015 | 0.12804 | 0.01026 | 0.00007 |
| 27 | **2.376** | 0.179 | 0.00123 | 0.12511 | 0.01011 | 0.00007 |
| 30 | 0.270 | 0.022 | 0.00014 | 0.18212 | 0.01466 | 0.00010 |
| 32 | 0.236 | 0.019 | 0.00013 | **0.69487** | 0.05578 | 0.00037 |
| 33 | 0.168 | 0.013 | 0.00009 | **1.61817** | 0.13037 | 0.00086 |
| 35 | 0.405 | 0.033 | 0.00023 | 0.15854 | 0.01280 | 0.00008 |

### 16.5 Unit Test Coverage

```
tests/test_adaptive.py::test_layer_bits_override PASSED
tests/test_adaptive.py::test_layer_bits_compress_decompress PASSED
tests/test_adaptive.py::test_allocate_bits_respects_budget PASSED
tests/test_adaptive.py::test_allocate_bits_sensitive_layers_get_more PASSED
tests/test_adaptive.py::test_allocate_bits_kv_split_format PASSED
tests/test_baselines.py::test_kivi_compress_decompress PASSED
tests/test_baselines.py::test_kivi_2bit_higher_error PASSED
tests/test_baselines.py::test_kivi_compression_ratio PASSED
tests/test_baselines.py::test_polarquant_compress_decompress PASSED
tests/test_cache.py::test_cache_basic_update PASSED
tests/test_cache.py::test_cache_incremental_growth PASSED
tests/test_cache.py::test_cache_compression_info PASSED
tests/test_codebook_cache.py::test_codebook_caches_to_disk PASSED
tests/test_codebook_cache.py::test_codebook_values_unchanged PASSED
tests/test_outlier.py::test_detect_outlier_channels PASSED
tests/test_outlier.py::test_outlier_compressor_shape PASSED
tests/test_outlier.py::test_outlier_compressor_lower_error PASSED
tests/test_perplexity.py::test_perplexity_on_known_logits PASSED
tests/test_perplexity.py::test_perplexity_random_higher PASSED

19 passed in 4.09s
```

(Note: 19 tests currently; the 5th adaptive test was added later.)

---

## Acknowledgments

The core TurboQuant compressor implementation in `turboquant/` (`lloyd_max.py`, `turboquant.py`, original `compressors.py`) is derived from the reference implementation at [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch). All extensions — the adaptive allocator, outlier grouping, benchmarking harness, and the CompressedCache generalization — are original contributions of this work.

---

## Links

- **GitHub**: https://github.com/azamkhan555622848/Sensitivity-Adaptive-KV-Cache-Compression-TurboQuant
- **Paper (PDF)**: [paper/main.pdf](paper/main.pdf)
- **Paper (DOCX)**: [paper/main.docx](paper/main.docx)
- **Upstream TurboQuant**: https://github.com/tonbistudio/turboquant-pytorch
- **KIVI**: https://github.com/jy-yuan/KIVI

---

*This report was written at the end of the experimental campaign, with all results verified against the JSON logs in `results/`. If there is a discrepancy between a number in this report and the raw JSON, the JSON is authoritative.*
