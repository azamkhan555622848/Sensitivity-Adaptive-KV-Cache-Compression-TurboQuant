# TurboQuant Research Paper

## Project
Conference paper on TurboQuant KV cache compression with three contributions:
1. Characterization of GQA amplification failure mode for VQ-based KV compression
2. Sensitivity-adaptive per-layer bit allocation (adaptive DP)
3. Outlier-aware channel grouping for high-GQA models

Targeting top-tier venue (ICML/NeurIPS or systems journal) after reviewer response revision.

## Phase 1 (Initial Implementation) — DONE
- [x] Step 1: Core infrastructure & baselines
- [x] Step 2: Mistral-7B perplexity sweep
- [x] Step 3: Qwen2.5-3B perplexity + calibration
- [x] Step 4: Needle-in-haystack on Mistral
- [x] Step 5: Adaptive allocation comparison
- [x] Step 6: Outlier-aware channel grouping experiments
- [x] Step 7: Multi-budget adaptive on Qwen
- [x] Step 8: Llama-3.1-8B sweep + Gemma-3-27B sweep + Llama-3.1-70B FP16 only (OOM)
- [x] Step 9: Memory/latency benchmarks (4 models)
- [x] Step 10: Initial paper draft (10 pages, PDF + DOCX)

## Phase 2 (Reviewer Response Revision) — IN PROGRESS

Response to reviewer feedback (see `/home/sirapop/.claude/plans/lexical-humming-mochi.md` for full plan).

- [x] **P1**: Qwen2.5-32B replaces Llama-3.1-70B (Nemotron 49B incompatible with transformers 5.5.0)
  - [x] Verify architecture: GQA 5:1, 64 layers, head_dim=128
  - [x] Full perplexity sweep: FP16=3.97, TQ K4V4=4.03 (+1.4%), KIVI-4=4.20 (+5.8%)
  - [x] Calibration + outlier profiling: avg K=4.38 V=3.62; only 2 outlier channels on layer 0
  - [x] Adaptive/outlier rescue: not needed at GQA 5:1 (uniform K4V4 already near-lossless)
  - [ ] Qwen32B hardware benchmarks (running)
  - [ ] Qwen32B downstream tasks (queued)
- [x] **P2**: Fix Gemma-3 calibration NaN
  - [x] Rewrote `calibrate()` with k_proj/v_proj hooks (also handles Gemma-3 language_model nesting)
  - [x] Re-run Gemma-3 calibration: all 62 layers, no NaN, avg K=4.56 V=3.44
- [x] **P3**: Expand downstream evaluation
  - [x] Fixed `eval/downstream.py` with CompressedHFLM subclass
  - [x] Mistral-7B downstream: FP16/TQ/KIVI identical on MMLU=0.608, ARC=0.590, HellaSwag=0.550, WinoGrande=0.780
  - [x] LongBench-E runner written (`eval/longbench.py`) — sweeps deferred
  - [ ] Multi-dataset perplexity — deferred (WikiText-2 results comprehensive)
- [x] **P4**: Statistical rigor
  - [x] `seed` parameter added to calibrate() + run_multiseed.py written
  - [x] Calibration sample count ablation: PPL ±0.016 across 4-64 samples — highly stable
- [x] **P5**: Ablations
  - [x] Calibration sample count: converges at n=4, stable through n=64
  - [ ] Calibration domain (WikiText-2, C4, The Stack) — script written, deferred
  - [x] Outlier threshold (3, 5, 7, 10): PPL varies only 0.06 across 3x range — robust
- [x] **P6**: Formal theory section
  - [x] Derived GQA amplification bound: O(g) vs O(√g), crossover condition
  - [x] Synthetic GQA experiment: TQ advantage decreases with g (matches theory)
  - [x] New paper Section 5: Theory integrated
- [x] **P7**: Paper revision
  - [x] Added Qwen2.5-32B to main PPL table
  - [x] Downstream results table (Mistral, all 3 methods identical)
  - [ ] Trim TurboQuant background (14 pages → target 12)
  - [x] Tightened language ("catastrophically" → "degrades sharply", etc.)
  - [x] Added explicit `tonbistudio/turboquant-pytorch` acknowledgment
- [x] **P8**: Killer figure + roofline analysis
  - [x] GQA ratio vs PPL degradation plot (Figure 1 in paper)
  - [x] Roofline latency projection table (paper/tables/roofline.tex)
- [x] **P9**: Update CLAUDE.md (this update)
- [x] **P10**: Git push — all commits on main branch, GitHub up to date

## Server
- Host: asus@140.113.202.36 (SSH key auth, no password needed)
- GPUs: 3x NVIDIA RTX A6000 (48GB each = 144GB)
- Venv: ~/TurboQuant-Research/venv
- Workflow: rsync code -> run on server -> rsync results back
- transformers v5.5.0: DynamicCache uses `.layers[i].keys/.values`
- HF token set on server (supports gated Llama-3.1, Gemma-3, Nemotron-49B models)

## Key Results (Phase 1, WikiText-2 PPL)

| Model | GQA | FP16 | TQ K4V4 | KIVI-4 | Best rescue |
|-------|-----|------|---------|--------|-------------|
| Gemma-3-27B | 2:1 | 7.47 | 7.52 (+0.7%) | 9.14 (+22%) | uniform already best |
| Mistral-7B | 4:1 | 4.94 | 4.96 (+0.4%) | 5.13 (+3.9%) | uniform already best |
| Llama-3.1-8B | 4:1 | 6.48 | 6.58 (+1.6%) | 7.29 (+12%) | uniform already best |
| Qwen2.5-3B | 8:1 | 7.63 | **42.33 COLLAPSE** | 7.89 (+3.4%) | **Outlier K4V4: 7.74 (+1.5%)** |
| Llama-3.1-70B | 8:1 | 3.40 | OOM | OOM | — (replaced by Nemotron 49B in Phase 2) |

Key findings:
- TurboQuant's advantage over KIVI scales inversely with GQA ratio
- Keys 10-100x more sensitive than values across all models
- Outlier-aware rescues Qwen from collapse (42 → 7.74 PPL)
- GQA 2:1 models: TQ is 30x closer to FP16 than KIVI
- GQA 8:1 models: uniform TQ fails, needs adaptive/outlier

## Phase 1 Hardware Benchmarks (prompt=2048, gen=32-64)

| Method | Mistral-7B | Llama-8B | Qwen-3B | Gemma-3-27B |
|--------|-----------|----------|---------|-------------|
| FP16 tok/s | 37.7 | 16.0 | 24.4 | 5.5 |
| TQ K4V4 tok/s | 3.8 (10x) | 2.1 (8x) | 2.4 (10x) | 1.6 (3.4x) |
| KIVI-4 tok/s | 7.7 (5x) | 7.7 (2x) | 7.8 (3x) | 4.4 (1.3x) |

Python-level implementation is 3-10x slower than FP16. Phase 2 will add roofline analysis showing theoretical 4-8x speedups achievable with fused kernels.

## Paper
- Location: `paper/main.tex` (LaTeX) → `paper/main.pdf` (10 pages) + `paper/main.docx`
- Build: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
- DOCX: `python scripts/build_docx.py`
- Phase 2 target: ≤ 12 pages with new theory section, downstream results, killer figure
- GitHub: https://github.com/azamkhan555622848/Sensitivity-Adaptive-KV-Cache-Compression-TurboQuant (main branch)

## Key References
- Core TurboQuant code derived from [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
- KIVI: https://github.com/jy-yuan/KIVI
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- LongBench: THUDM/LongBench dataset on HuggingFace
