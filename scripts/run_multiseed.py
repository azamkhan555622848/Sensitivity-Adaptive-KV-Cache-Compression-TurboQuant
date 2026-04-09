#!/usr/bin/env python3
"""Multi-seed wrapper for calibration-dependent methods.

Runs calibration + outlier profiling + perplexity evaluation for multiple
seeds, then aggregates the results into mean ± std. Only calibration-
dependent methods (adaptive, outlier) are seed-sensitive; uniform methods
(FP16, TQ-V3 K4V4, KIVI-4) are deterministic and run once.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from eval.perplexity import evaluate_perplexity
from turboquant.adaptive import calibrate, allocate_bits
from turboquant.outlier import detect_outlier_channels
from turboquant.compressors_v3 import MSECompressor
from turboquant.baselines.registry import create_cache


def _sensitivity_from_kv(kv_data, bit_options=(2, 3, 4, 6, 8)):
    sensitivity = {}
    head_dim = next(iter(kv_data.values()))["keys"].shape[-1]
    for layer_idx, data in kv_data.items():
        sensitivity[layer_idx] = {"key": {}, "value": {}}
        k_orig = data["keys"]
        v_orig = data["values"]
        for bits in bit_options:
            comp = MSECompressor(head_dim, bits, seed=42 + layer_idx * 1000, device="cpu")
            k_q = comp.decompress(comp.compress(k_orig))
            v_q = comp.decompress(comp.compress(v_orig))
            sensitivity[layer_idx]["key"][bits] = (k_orig - k_q).pow(2).mean().item()
            sensitivity[layer_idx]["value"][bits] = (v_orig - v_q).pow(2).mean().item()
    return sensitivity


def _outliers_from_kv(kv_data, threshold=5.0):
    layer_outliers = {}
    for layer_idx, data in kv_data.items():
        k_mask = detect_outlier_channels(data["keys"], threshold_factor=threshold)
        v_mask = detect_outlier_channels(data["values"], threshold_factor=threshold)
        layer_outliers[layer_idx] = {
            "key_outlier_channels": k_mask.nonzero(as_tuple=True)[0].tolist(),
            "value_outlier_channels": v_mask.nonzero(as_tuple=True)[0].tolist(),
            "key_outlier_count": int(k_mask.sum().item()),
            "value_outlier_count": int(v_mask.sum().item()),
        }
    return layer_outliers


def run_one_seed(model, tokenizer, model_info, seed, method, budget=4.0,
                  max_seq_len=512, n_samples=16, max_tokens=50000):
    """Run calibration + one method + perplexity measurement for one seed."""
    print(f"  Calibrating (seed={seed})...")
    kv_data = calibrate(model, tokenizer, n_samples=n_samples,
                         max_seq_len=max_seq_len, seed=seed)

    if method == "adaptive":
        sens = _sensitivity_from_kv(kv_data)
        allocation = allocate_bits(sens, budget=budget)
        method_config = {
            "type": "turboquant-adaptive",
            "params": {
                "budget": budget,
                "residual_window": 128,
                "_layer_allocation": allocation,
            },
        }
    elif method == "outlier":
        outliers = _outliers_from_kv(kv_data, threshold=5.0)
        tmp = f"/tmp/_multiseed_outlier_{seed}.json"
        with open(tmp, "w") as f:
            json.dump({"layer_outliers": {str(k): v for k, v in outliers.items()}}, f)
        method_config = {
            "type": "turboquant-outlier",
            "params": {
                "key_bits": 4,
                "value_bits": 4,
                "residual_window": 128,
                "outlier_file": tmp,
            },
        }
    else:
        raise ValueError(f"Unsupported method for multi-seed: {method}")

    def factory():
        return create_cache(method_config, model_info)

    result = evaluate_perplexity(
        model, tokenizer, cache_factory=factory,
        dataset_name="wikitext2", max_seq_len=2048, max_tokens=max_tokens,
    )
    return result["perplexity"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["adaptive", "outlier"], default="outlier")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--output-dir", default="results/multiseed")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    safe = args.model.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"{safe}_{args.method}_multiseed.json")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)
    model_info = get_model_info(model)

    ppls = []
    per_seed = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        t0 = time.time()
        ppl = run_one_seed(
            model, tokenizer, model_info, seed, args.method,
            budget=args.budget, max_seq_len=args.max_seq_len,
            n_samples=args.n_samples, max_tokens=args.max_tokens,
        )
        elapsed = time.time() - t0
        print(f"  seed={seed}: PPL={ppl:.4f} ({elapsed:.0f}s)")
        ppls.append(ppl)
        per_seed.append({"seed": seed, "ppl": ppl, "elapsed_sec": elapsed})

    mean = sum(ppls) / len(ppls)
    var = sum((p - mean) ** 2 for p in ppls) / max(len(ppls) - 1, 1)
    std = math.sqrt(var)
    print(f"\n=== Summary ===")
    print(f"Model: {args.model}")
    print(f"Method: {args.method} (budget={args.budget})")
    print(f"Seeds: {args.seeds}")
    print(f"PPL: {mean:.4f} ± {std:.4f}")
    print(f"Individual: {ppls}")

    result = {
        "model": args.model,
        "method": args.method,
        "budget": args.budget,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "mean_ppl": mean,
        "std_ppl": std,
        "min_ppl": min(ppls),
        "max_ppl": max(ppls),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
