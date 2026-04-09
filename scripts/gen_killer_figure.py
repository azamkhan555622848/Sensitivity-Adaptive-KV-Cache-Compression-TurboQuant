#!/usr/bin/env python3
"""Generate the 'killer figure': GQA ratio vs PPL degradation.

Loads all perplexity results across models, aggregates them by (model, method),
and produces a scatter + line plot showing how each method's relative PPL
degradation scales with the model's GQA ratio.

Output: paper/figures/killer.pdf (and .png)
"""
import glob
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy required. Install: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(os.path.join(PAPER_DIR, "figures"), exist_ok=True)


MODEL_META = {
    "gemma-3-27b-it": ("Gemma-3-27B", 2),
    "Mistral-7B-Instruct-v0.3": ("Mistral-7B", 4),
    "Llama-3.1-8B-Instruct": ("Llama-3.1-8B", 4),
    "Qwen2.5-3B-Instruct": ("Qwen-3B", 8),
    "Qwen2.5-32B-Instruct": ("Qwen-32B", 5),
}

# For Qwen-3B, use the adaptive+outlier result rather than the collapsed uniform.


def load_all_ppl():
    """Walk results/*/*.json (excluding benchmarks/calibration/outlier) and
    return {(model_key, method_key): ppl}."""
    by_mm = {}
    for sweep in glob.glob(os.path.join(RESULTS_DIR, "*")):
        if not os.path.isdir(sweep):
            continue
        name = os.path.basename(sweep)
        if name in ("benchmarks", "calibration", "outlier"):
            continue
        for f in glob.glob(os.path.join(sweep, "*.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if d.get("benchmark") != "perplexity":
                continue
            try:
                ppl = d["results"]["wikitext2"]["perplexity"]
            except (KeyError, TypeError):
                continue
            model_key = d["model"].split("/")[-1]
            method = d["method"]
            mc = d.get("method_config", {})
            # Build a method key
            if method == "fp16":
                mk = "fp16"
            elif method == "turboquant-v3":
                kb = mc.get("key_bits", 4)
                vb = mc.get("value_bits", 4)
                rw = mc.get("residual_window", 128)
                if rw != 128:
                    continue
                mk = f"tq_k{kb}v{vb}"
            elif method == "kivi":
                mk = f"kivi{mc.get('bits', 4)}"
            elif method == "turboquant-adaptive":
                mk = f"adaptive_{mc.get('budget', 4.0)}"
            elif method == "turboquant-outlier":
                mk = "outlier"
            elif method == "turboquant-adaptive-outlier":
                mk = "adaptive_outlier"
            else:
                continue
            key = (model_key, mk)
            if key not in by_mm:
                by_mm[key] = ppl
    return by_mm


def get_data():
    """Return a dict {method_label: [(gqa, pct_degradation, model_key)]}."""
    ppl_by_mm = load_all_ppl()

    method_labels = {
        "tq_k4v4": "TurboQuant K4V4 (uniform)",
        "kivi4": "KIVI-4",
        "adaptive_4.0": "TurboQuant-Adaptive (b=4)",
        "outlier": "TurboQuant-Outlier K4V4",
    }
    data = {lbl: [] for lbl in method_labels.values()}

    for model_key, (display, gqa) in MODEL_META.items():
        fp16 = ppl_by_mm.get((model_key, "fp16"))
        if fp16 is None:
            print(f"Skip {model_key}: no FP16", file=sys.stderr)
            continue
        for mk, lbl in method_labels.items():
            ppl = ppl_by_mm.get((model_key, mk))
            if ppl is None:
                continue
            pct = (ppl - fp16) / fp16 * 100
            # clip collapses to 10000% for plotting
            pct = min(max(pct, 0.01), 10000)
            data[lbl].append((gqa, pct, display))
    return data, method_labels


def main():
    data, method_labels = get_data()
    fig, ax = plt.subplots(figsize=(7.5, 5))

    markers = {
        "TurboQuant K4V4 (uniform)": ("o", "tab:blue"),
        "KIVI-4": ("s", "tab:orange"),
        "TurboQuant-Adaptive (b=4)": ("^", "tab:green"),
        "TurboQuant-Outlier K4V4": ("D", "tab:red"),
    }

    for label, points in data.items():
        if not points:
            continue
        pts = sorted(points)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        marker, color = markers.get(label, ("o", "gray"))
        ax.plot(xs, ys, marker=marker, linestyle="-", color=color,
                markersize=9, linewidth=1.8, label=label)
        # Annotate each point with the model name (small text above)
        for gqa, pct, name in pts:
            ax.annotate(
                name,
                xy=(gqa, pct),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.85,
            )

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("GQA ratio (query heads per KV head)", fontsize=11)
    ax.set_ylabel("WikiText-2 PPL degradation vs FP16 (%)\n(log scale)", fontsize=11)
    ax.set_title(
        "TurboQuant's advantage scales inversely with GQA ratio",
        fontsize=12,
    )
    ax.set_xticks([2, 4, 5, 8])
    ax.set_xlim(1.5, 9)
    ax.set_ylim(0.05, 20000)
    ax.axhline(y=100, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(8.3, 120, "collapse\nthreshold", fontsize=8, color="red", ha="left")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out_pdf = os.path.join(PAPER_DIR, "figures", "killer.pdf")
    out_png = os.path.join(PAPER_DIR, "figures", "killer.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
