# Sensitivity-Adaptive KV Cache Compression

Research codebase for **"Sensitivity-Adaptive KV Cache Compression: Rescuing Vector Quantization from Grouped Query Attention"**, extending TurboQuant with sensitivity-adaptive per-layer bit allocation and outlier-aware channel grouping.

## Key Findings

We evaluate TurboQuant across 5 modern LLMs and identify **GQA amplification**: the effectiveness of vector-quantization KV cache compression scales inversely with the grouped query attention ratio.

| Model | GQA ratio | TQ K4V4 vs FP16 | KIVI-4 vs FP16 | TQ advantage |
|-------|-----------|----------------|----------------|--------------|
| Gemma-3-27B | 2:1 | **+0.7%** | +22.4% | **30x** |
| Mistral-7B | 4:1 | +0.4% | +3.9% | 9x |
| Llama-3.1-8B | 4:1 | +1.6% | +12.4% | 8x |
| Qwen2.5-3B | 8:1 | collapse (+455%) | +3.4% | KIVI wins |

For high-GQA models where uniform TurboQuant collapses, our contributions rescue it:

- **Outlier-aware channel grouping** on Qwen2.5-3B reduces 4-bit perplexity from **42.33 → 7.74** (within 1.5% of FP16)
- **Sensitivity-adaptive allocation** extends usable compression below 3 bits via dynamic programming

## Repository Structure

```
TurboQuant-Research/
  turboquant/              # Core library
    compressors_v3.py      #   TurboQuant-V3 compressor (normalize → rotate → Lloyd-Max)
    lloyd_max.py           #   Optimal Lloyd-Max codebook for Beta distribution
    cache.py               #   CompressedCache (DynamicCache subclass)
    adaptive.py            #   Sensitivity profiling + DP bit allocation
    outlier.py             #   Outlier-aware channel grouping
    baselines/
      kivi.py              #   KIVI per-channel asymmetric quantization
      polarquant.py        #   PolarQuant baseline
      registry.py          #   Method factory registry
  eval/                    # Evaluation harnesses
    perplexity.py          #   WikiText-2 perplexity
    needle.py              #   Needle-in-a-haystack
    metrics.py             #   Latency and memory measurement
    runner.py              #   Experiment orchestration
  configs/
    models/                # Per-model configs
    sweeps/                # Experiment sweeps (YAML)
  scripts/
    run_experiment.py      # Main experiment runner
    calibrate.py           # Sensitivity calibration
    profile_outliers.py    # Outlier channel detection
    benchmark_hw.py        # Hardware benchmarks
    gen_paper_tables.py    # Auto-generate LaTeX tables from results
    gen_sensitivity_plot.py# Auto-generate sensitivity figure
  tests/                   # Unit tests (18 tests, all passing)
  results/                 # Experiment outputs (JSON)
  paper/                   # LaTeX paper source
    main.tex               # Main paper
    references.bib         # Bibliography
    tables/                # Auto-generated tables
    figures/               # Auto-generated figures
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scipy transformers datasets accelerate pyyaml pandas lm-eval rouge-score matplotlib pytest
```

### Run Tests

```bash
pytest tests/ -v
```

### Run a Quick Experiment

```bash
python scripts/run_experiment.py --config configs/sweeps/quick-test.yaml
```

### Profile Sensitivity and Allocate Bits

```bash
python scripts/calibrate.py --model Qwen/Qwen2.5-3B-Instruct --budget 4.0
```

### Profile Outlier Channels

```bash
python scripts/profile_outliers.py --model Qwen/Qwen2.5-3B-Instruct
```

### Build the Paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Acknowledgments

The core TurboQuant compressor implementation in `turboquant/` (particularly `lloyd_max.py`, `turboquant.py`, and the original `compressors.py`) is derived from the reference implementation at [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch). Our extensions include:

- **`compressors_v3.py`** — MSE-only variant, asymmetric key/value bit-widths, residual window, per-layer bit overrides
- **`cache.py`** — generalized `CompressedCache` (HuggingFace `DynamicCache` subclass) with pluggable compressor backends
- **`adaptive.py`** — sensitivity profiling and dynamic-programming bit allocation (novel contribution)
- **`outlier.py`** — outlier-aware channel grouping (novel contribution)
- **`baselines/`** — KIVI and PolarQuant baselines for comparison
- **`eval/`, `scripts/`, `configs/`** — evaluation harness, experiment runner, benchmark sweeps

## Cited Methods

- **TurboQuant** (ICLR 2026): the base vector-quantization compressor ([tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch))
- **KIVI** (ICML 2024): per-channel asymmetric scalar quantization baseline
- **PolarQuant**: polar decomposition variant baseline
- **KVQuant** (NeurIPS 2024): outlier-aware scalar quantization

## Citation

If you use this codebase, please cite:

```bibtex
@misc{turboquant-adaptive2026,
  title={Sensitivity-Adaptive KV Cache Compression: Rescuing Vector Quantization from Grouped Query Attention},
  author={Anonymous},
  year={2026},
  note={\url{https://github.com/azamkhan555622848/Sensitivity-Adaptive-KV-Cache-Compression-TurboQuant}}
}
```
