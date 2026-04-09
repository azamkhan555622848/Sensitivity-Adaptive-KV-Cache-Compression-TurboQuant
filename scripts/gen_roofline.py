#!/usr/bin/env python3
"""Roofline latency projection for compressed KV cache.

Builds the honest system-level story for the paper: the current Python
reference implementation is slow, but the theoretical bandwidth savings
give a specific projected speedup that a fused-kernel implementation would
realize. We present:

1. Current measured decode throughput (from benchmarks/).
2. Theoretical KV cache storage per method (bits/element).
3. Theoretical bandwidth savings during decode (dominated by KV cache reads).
4. Projected fused-kernel decode throughput assuming decode is memory-
   bandwidth-bound (which is the standard assumption in the kernel-
   quantization literature).

Outputs a LaTeX table to paper/tables/roofline.tex.
"""
import glob
import json
import os
import sys

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(os.path.join(PAPER_DIR, "tables"), exist_ok=True)

# Method bit budgets. For TQ-*, this is K+V bits averaged. For KIVI, just the
# single bit-width (KIVI has small metadata overhead ~0.5 bits at group_size=128
# which we fold in as 0.5 extra bits per element).
METHOD_BITS = {
    "fp16": 16.0,
    "tq_k4v4": 4.0,
    "tq_k4v2": 3.0,
    "tq_k2v2": 2.0,
    "kivi_4": 4.5,  # KIVI 4-bit + ~0.5 bit per element for per-channel scale/zero
    "kivi_2": 2.5,
}

METHOD_LABELS = {
    "fp16": "FP16",
    "tq_k4v4": "TQ K4V4",
    "tq_k4v2": "TQ K4V2",
    "tq_k2v2": "TQ K2V2",
    "kivi_4": "KIVI-4",
    "kivi_2": "KIVI-2",
}

MODEL_ORDER = [
    ("mistralai_Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("meta-llama_Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen_Qwen2.5-3B-Instruct", "Qwen-3B"),
    ("google_gemma-3-27b-it", "Gemma-3-27B"),
]


def load_benchmarks():
    benchmarks = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "benchmarks", "*.json")):
        with open(f) as fh:
            data = json.load(fh)
        model = data["model"].split("/")[-1]
        # Normalize key: replace / with _
        key = data["model"].replace("/", "_")
        benchmarks[key] = data["results"]
    return benchmarks


def project_decode(fp16_tps, method_bits):
    """Given FP16 decode tok/s, project fused-kernel decode tok/s for a method
    that uses method_bits bits per KV element. Assumes decode is memory-
    bandwidth bound on the KV cache, so speedup ≈ 16 / method_bits."""
    if method_bits <= 0:
        return fp16_tps
    speedup = 16.0 / method_bits
    return fp16_tps * speedup


def main():
    benchmarks = load_benchmarks()

    lines = []
    lines.append("\\begin{tabular}{l l c c c c}")
    lines.append("\\toprule")
    lines.append("Model & Method & Bits/elem & Measured tok/s & Projected tok/s & Compression \\\\")
    lines.append("\\midrule")

    for model_key, display in MODEL_ORDER:
        res = benchmarks.get(model_key, {})
        if not res:
            continue
        fp16_block = res.get("fp16", {})
        fp16_tps = fp16_block.get("latency", {}).get("decode_tokens_per_sec")
        if fp16_tps is None:
            continue

        for mk, label in METHOD_LABELS.items():
            block = res.get(mk, {})
            lat = block.get("latency", {})
            measured = lat.get("decode_tokens_per_sec")
            bits = METHOD_BITS[mk]
            compression_ratio = 16.0 / bits
            projected = project_decode(fp16_tps, bits)
            measured_str = f"{measured:.1f}" if isinstance(measured, (int, float)) else "--"
            projected_str = f"{projected:.1f}" if mk != "fp16" else "--"
            compression_str = f"{compression_ratio:.2f}$\\times$"
            model_cell = display if mk == "fp16" else ""
            lines.append(
                f"{model_cell} & {label} & {bits:.1f} & {measured_str} & {projected_str} & {compression_str} \\\\"
            )
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(PAPER_DIR, "tables", "roofline.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
