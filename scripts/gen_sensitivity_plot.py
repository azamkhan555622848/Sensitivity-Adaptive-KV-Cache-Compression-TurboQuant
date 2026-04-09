#!/usr/bin/env python3
"""Generate the sensitivity analysis figure for the paper."""
import json, os, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(os.path.join(PAPER_DIR, "figures"), exist_ok=True)

def load_calibration(path):
    with open(path) as f:
        data = json.load(f)
    sens = {}
    for k, v in data["sensitivity"].items():
        layer = int(k)
        if "key" in v and "value" in v:
            sens[layer] = {
                "key": {int(b): mse for b, mse in v["key"].items()},
                "value": {int(b): mse for b, mse in v["value"].items()},
            }
        else:
            # legacy format with combined
            sens[layer] = {"combined": {int(b): mse for b, mse in v.items()}}
    alloc = {int(k): tuple(v) for k, v in data.get("allocation", {}).items()}
    return sens, alloc

def main():
    mistral_path = os.path.join(RESULTS_DIR, "calibration", "mistralai_Mistral-7B-Instruct-v0.3_budget4.0_kvsplit.json")
    qwen_path = os.path.join(RESULTS_DIR, "calibration", "Qwen_Qwen2.5-3B-Instruct_budget4.0.json")

    mistral_sens, _ = load_calibration(mistral_path)
    qwen_sens, _ = load_calibration(qwen_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # Left panel: sensitivity curves at 4-bit
    ax = axes[0]
    mistral_layers = sorted(mistral_sens.keys())
    qwen_layers = sorted(qwen_sens.keys())

    mistral_k = [mistral_sens[l]["key"][4] for l in mistral_layers]
    mistral_v = [mistral_sens[l]["value"][4] for l in mistral_layers]
    qwen_k = [qwen_sens[l]["key"][4] for l in qwen_layers]
    qwen_v = [qwen_sens[l]["value"][4] for l in qwen_layers]

    ax.semilogy(mistral_layers, mistral_k, "o-", color="tab:blue", label="Mistral-7B keys", markersize=4)
    ax.semilogy(mistral_layers, mistral_v, "o--", color="tab:blue", label="Mistral-7B values", markersize=4, alpha=0.6)
    ax.semilogy(qwen_layers, qwen_k, "s-", color="tab:red", label="Qwen2.5-3B keys", markersize=4)
    ax.semilogy(qwen_layers, qwen_v, "s--", color="tab:red", label="Qwen2.5-3B values", markersize=4, alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("4-bit MSE distortion (log scale)")
    ax.set_title("Per-layer sensitivity: keys vs values")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Right panel: optimal allocations for Qwen at different budgets
    ax = axes[1]
    budgets = [2.5, 3.0, 4.0]
    colors = ["tab:green", "tab:orange", "tab:purple"]

    for budget, color in zip(budgets, colors):
        alloc_path = os.path.join(
            RESULTS_DIR, "calibration",
            f"Qwen_Qwen2.5-3B-Instruct_budget{budget}_kvsplit.json"
        )
        if not os.path.exists(alloc_path):
            continue
        with open(alloc_path) as f:
            d = json.load(f)
        alloc = {int(k): tuple(v) for k, v in d["allocation"].items()}
        layers = sorted(alloc.keys())
        key_bits = [alloc[l][0] for l in layers]
        value_bits = [alloc[l][1] for l in layers]
        ax.plot(layers, key_bits, "-", color=color, label=f"K, $\\bar b$={budget}", linewidth=1.8)
        ax.plot(layers, value_bits, "--", color=color, label=f"V, $\\bar b$={budget}", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Layer index (Qwen2.5-3B)")
    ax.set_ylabel("Allocated bits")
    ax.set_title("Adaptive bit allocation (Qwen2.5-3B)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 9)

    fig.tight_layout()
    out_pdf = os.path.join(PAPER_DIR, "figures", "sensitivity.pdf")
    out_png = os.path.join(PAPER_DIR, "figures", "sensitivity.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()
