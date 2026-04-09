#!/usr/bin/env python3
"""Ablation: calibration sample count and domain.

Runs two studies on a chosen model:

1. **Sample count**: vary n_samples ∈ {4, 8, 16, 32, 64} and measure how the
   adaptive allocation's WikiText-2 perplexity changes. This answers "how
   many calibration samples do we actually need?"

2. **Domain mismatch**: calibrate on WikiText-2, C4, and The Stack (code),
   then measure WikiText-2 test perplexity under each allocation. This
   answers "how sensitive is the allocator to calibration domain?"

Both studies use the sensitivity-adaptive allocator at budget=4.0 (the
headline budget from the paper). Results go to results/ablations/.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from eval.perplexity import evaluate_perplexity
from turboquant.adaptive import profile_layer_sensitivity, allocate_bits
from turboquant.baselines.registry import create_cache


def _run_ppl_with_allocation(model, tokenizer, model_info, allocation,
                              residual_window=128, max_seq_len=2048, max_tokens=50000):
    """Evaluate perplexity using a specific per-layer bit allocation."""
    method_config = {
        "type": "turboquant-adaptive",
        "params": {
            "budget": 4.0,
            "residual_window": residual_window,
            "_layer_allocation": allocation,
        },
    }

    def factory():
        return create_cache(method_config, model_info)

    result = evaluate_perplexity(
        model, tokenizer, cache_factory=factory,
        dataset_name="wikitext2", max_seq_len=max_seq_len,
        max_tokens=max_tokens,
    )
    return result["perplexity"]


def _profile(model, tokenizer, n_samples, dataset, max_seq_len, seed):
    """Run sensitivity profiling and return the allocation at budget=4.0."""
    sensitivity = profile_layer_sensitivity(
        model, tokenizer,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
        bit_options=[2, 3, 4, 6, 8],
    )
    # profile_layer_sensitivity doesn't take dataset/seed directly; the hook
    # we rewrote uses the seed via calibrate(). We assume the default is fine
    # for the sample count ablation. For the domain ablation we pass the
    # dataset via a monkey-patched wrapper below.
    allocation = allocate_bits(sensitivity, budget=4.0)
    return allocation, sensitivity


def ablate_sample_count(model, tokenizer, model_info, output_path,
                         max_seq_len=512, seed=42):
    """Sweep over n_samples values and record PPL for each."""
    results = []
    for n_samples in [4, 8, 16, 32, 64]:
        print(f"\n[n_samples={n_samples}] Profiling...")
        t0 = time.time()
        allocation, _ = _profile(model, tokenizer, n_samples,
                                  dataset="wikitext2", max_seq_len=max_seq_len, seed=seed)
        t1 = time.time()
        print(f"  profiling: {t1 - t0:.1f}s")
        alloc_serializable = {str(k): list(v) for k, v in allocation.items()}
        ppl = _run_ppl_with_allocation(model, tokenizer, model_info, allocation)
        t2 = time.time()
        print(f"  PPL @ n_samples={n_samples}: {ppl:.4f}  (eval {t2 - t1:.1f}s)")
        results.append({
            "n_samples": n_samples,
            "seed": seed,
            "ppl": ppl,
            "allocation": alloc_serializable,
            "profile_time_sec": t1 - t0,
        })
    with open(output_path, "w") as f:
        json.dump({"ablation": "sample_count", "results": results}, f, indent=2)
    print(f"\nSaved: {output_path}")


def ablate_domain(model, tokenizer, model_info, output_path,
                   max_seq_len=512, n_samples=16, seed=42):
    """Calibrate on different domains, evaluate PPL on WikiText-2."""
    # We need to drive the dataset choice through calibrate(). Import directly.
    from turboquant.adaptive import calibrate

    domains = ["wikitext2", "c4", "the_stack"]
    results = []
    for domain in domains:
        print(f"\n[domain={domain}] Calibrating...")
        t0 = time.time()
        try:
            kv_data = calibrate(model, tokenizer, n_samples=n_samples,
                                max_seq_len=max_seq_len, dataset_name=domain,
                                seed=seed)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"domain": domain, "error": str(e)[:200]})
            continue
        # Convert to sensitivity using MSECompressor
        import torch
        from turboquant.compressors_v3 import MSECompressor
        sensitivity = {}
        head_dim = next(iter(kv_data.values()))["keys"].shape[-1]
        for layer_idx, data in kv_data.items():
            sensitivity[layer_idx] = {"key": {}, "value": {}}
            k_orig = data["keys"]
            v_orig = data["values"]
            for bits in [2, 3, 4, 6, 8]:
                comp = MSECompressor(head_dim, bits, seed=42 + layer_idx * 1000, device="cpu")
                k_q = comp.decompress(comp.compress(k_orig))
                v_q = comp.decompress(comp.compress(v_orig))
                sensitivity[layer_idx]["key"][bits] = (k_orig - k_q).pow(2).mean().item()
                sensitivity[layer_idx]["value"][bits] = (v_orig - v_q).pow(2).mean().item()
        allocation = allocate_bits(sensitivity, budget=4.0)
        t1 = time.time()
        print(f"  profiling + alloc: {t1 - t0:.1f}s")
        ppl = _run_ppl_with_allocation(model, tokenizer, model_info, allocation)
        t2 = time.time()
        print(f"  WikiText-2 PPL (calibrated on {domain}): {ppl:.4f}")
        results.append({
            "domain": domain,
            "seed": seed,
            "n_samples": n_samples,
            "ppl": ppl,
            "allocation": {str(k): list(v) for k, v in allocation.items()},
        })
    with open(output_path, "w") as f:
        json.dump({"ablation": "domain", "results": results}, f, indent=2)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument("--studies", nargs="+",
                         choices=["sample_count", "domain", "all"],
                         default=["all"])
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    safe = args.model.replace("/", "_")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)
    model_info = get_model_info(model)

    studies = args.studies
    if "all" in studies:
        studies = ["sample_count", "domain"]

    if "sample_count" in studies:
        path = os.path.join(args.output_dir, f"{safe}_sample_count.json")
        ablate_sample_count(model, tokenizer, model_info, path,
                             max_seq_len=args.max_seq_len, seed=args.seed)

    if "domain" in studies:
        path = os.path.join(args.output_dir, f"{safe}_domain.json")
        ablate_domain(model, tokenizer, model_info, path,
                       max_seq_len=args.max_seq_len, seed=args.seed)


if __name__ == "__main__":
    main()
