#!/usr/bin/env python3
"""Ablation: outlier detection threshold.

Sweeps the outlier threshold_factor ∈ {3, 4, 5, 7, 10} on a chosen model
(typically Qwen2.5-3B where outliers matter) and reports:
- Total outlier channels per layer
- WikiText-2 perplexity under outlier-aware K4V4 compression

This answers the reviewer concern: "why threshold=5? How sensitive is the
method to this choice?"
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from eval.perplexity import evaluate_perplexity
from turboquant.adaptive import calibrate
from turboquant.outlier import detect_outlier_channels
from turboquant.baselines.registry import create_cache


def _profile_outliers(model, tokenizer, threshold, n_samples, max_seq_len):
    """Re-profile outliers at a given threshold and return the layer_outliers dict."""
    kv_data = calibrate(model, tokenizer, n_samples=n_samples, max_seq_len=max_seq_len)
    layer_outliers = {}
    for layer_idx, data in kv_data.items():
        k_mask = detect_outlier_channels(data["keys"], threshold_factor=threshold)
        v_mask = detect_outlier_channels(data["values"], threshold_factor=threshold)
        layer_outliers[layer_idx] = {
            "key_outlier_count": int(k_mask.sum().item()),
            "value_outlier_count": int(v_mask.sum().item()),
            "key_outlier_channels": k_mask.nonzero(as_tuple=True)[0].tolist(),
            "value_outlier_channels": v_mask.nonzero(as_tuple=True)[0].tolist(),
        }
    return layer_outliers


def _run_outlier_ppl(model, tokenizer, model_info, layer_outliers, max_tokens=50000):
    """Evaluate perplexity with outlier-aware K4V4 using the given outlier profile."""
    # Write outlier profile to a temp JSON since the factory loads from file
    tmp_path = "/tmp/_ablate_outlier_profile.json"
    with open(tmp_path, "w") as f:
        json.dump({"layer_outliers": {str(k): v for k, v in layer_outliers.items()}}, f)

    method_config = {
        "type": "turboquant-outlier",
        "params": {
            "key_bits": 4,
            "value_bits": 4,
            "residual_window": 128,
            "outlier_file": tmp_path,
        },
    }

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
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument("--thresholds", nargs="+", type=float,
                         default=[3.0, 4.0, 5.0, 7.0, 10.0])
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    safe = args.model.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"{safe}_outlier_threshold.json")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)
    model_info = get_model_info(model)

    results = []
    for thr in args.thresholds:
        print(f"\n[threshold={thr}] Profiling outliers...")
        t0 = time.time()
        layer_outliers = _profile_outliers(model, tokenizer, thr, args.n_samples, args.max_seq_len)
        t1 = time.time()
        total_k = sum(lo["key_outlier_count"] for lo in layer_outliers.values())
        total_v = sum(lo["value_outlier_count"] for lo in layer_outliers.values())
        max_k = max(lo["key_outlier_count"] for lo in layer_outliers.values()) if layer_outliers else 0
        print(f"  Total K outliers: {total_k}, V outliers: {total_v} (max per layer: {max_k})")
        print(f"  Profiling: {t1 - t0:.1f}s")

        ppl = _run_outlier_ppl(model, tokenizer, model_info, layer_outliers)
        t2 = time.time()
        print(f"  WikiText-2 PPL @ threshold={thr}: {ppl:.4f} (eval {t2 - t1:.1f}s)")

        results.append({
            "threshold": thr,
            "total_key_outliers": total_k,
            "total_value_outliers": total_v,
            "max_key_outliers_per_layer": max_k,
            "ppl": ppl,
            "n_layers": len(layer_outliers),
        })

    with open(out_path, "w") as f:
        json.dump({"ablation": "outlier_threshold", "results": results,
                   "model": args.model}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
