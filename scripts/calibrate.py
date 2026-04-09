#!/usr/bin/env python3
"""Run sensitivity profiling and optimal bit allocation for a model."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from turboquant.adaptive import profile_layer_sensitivity, allocate_bits

def main():
    parser = argparse.ArgumentParser(description="Calibrate per-layer bit allocation")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--budget", type=float, default=4.0, help="Average bits per component")
    parser.add_argument("--n-samples", type=int, default=16, help="Number of calibration samples")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for calibration")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        os.makedirs("results/calibration", exist_ok=True)
        args.output = f"results/calibration/{safe_name}_budget{args.budget}.json"

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    model_info = get_model_info(model)
    print(f"  n_layers={model_info['n_layers']}, head_dim={model_info['head_dim']}")

    print(f"\nProfiling layer sensitivity ({args.n_samples} samples, seq_len={args.max_seq_len})...")
    t0 = time.time()
    sensitivity = profile_layer_sensitivity(model, tokenizer, n_samples=args.n_samples,
                                            max_seq_len=args.max_seq_len)
    t1 = time.time()
    print(f"  Profiling took {t1-t0:.1f}s")

    # Print sensitivity summary (separate K/V)
    for component in ["key", "value"]:
        print(f"\nPer-layer {component.upper()} sensitivity (MSE distortion):")
        print(f"{'Layer':>6} | {'2-bit':>10} | {'3-bit':>10} | {'4-bit':>10} | {'6-bit':>10} | {'8-bit':>10}")
        print("-" * 72)
        for layer_idx in sorted(sensitivity.keys()):
            row = sensitivity[layer_idx][component]
            print(f"{layer_idx:>6} | {row.get(2, 0):>10.6f} | {row.get(3, 0):>10.6f} | "
                  f"{row.get(4, 0):>10.6f} | {row.get(6, 0):>10.6f} | {row.get(8, 0):>10.6f}")

    print(f"\nRunning DP bit allocation (budget={args.budget})...")
    allocation = allocate_bits(sensitivity, budget=args.budget)

    print(f"\nOptimal allocation:")
    print(f"{'Layer':>6} | {'Key bits':>8} | {'Value bits':>10}")
    print("-" * 35)
    total_k, total_v = 0, 0
    for layer_idx in sorted(allocation.keys()):
        kb, vb = allocation[layer_idx]
        total_k += kb
        total_v += vb
        print(f"{layer_idx:>6} | {kb:>8} | {vb:>10}")
    n = len(allocation)
    print(f"\n  Avg key bits: {total_k/n:.2f}, Avg value bits: {total_v/n:.2f}")
    print(f"  Total avg: {(total_k + total_v) / (2 * n):.2f} (budget: {args.budget})")

    # Save results
    result = {
        "model": args.model,
        "model_info": model_info,
        "budget": args.budget,
        "n_samples": args.n_samples,
        "max_seq_len": args.max_seq_len,
        "sensitivity": {str(k): v for k, v in sensitivity.items()},
        "allocation": {str(k): list(v) for k, v in allocation.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
