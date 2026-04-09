#!/usr/bin/env python3
"""Hardware benchmarks: memory and latency across methods."""
import sys, os, json, time, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from eval.model_loader import load_model, get_model_info
from eval.metrics import measure_latency, measure_memory
from turboquant.baselines.registry import create_cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--mem-seq-len", type=int, default=4096)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        os.makedirs("results/benchmarks", exist_ok=True)
        args.output = f"results/benchmarks/{safe_name}_hw.json"

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    info = get_model_info(model)
    print(f"  n_layers={info['n_layers']}, head_dim={info['head_dim']}, n_kv_heads={info['n_kv_heads']}")

    # Methods to test
    methods = [
        ("fp16", None),
        ("tq_k4v4", {"type": "turboquant-v3", "params": {"key_bits": 4, "value_bits": 4, "residual_window": 128}}),
        ("tq_k4v2", {"type": "turboquant-v3", "params": {"key_bits": 4, "value_bits": 2, "residual_window": 128}}),
        ("tq_k2v2", {"type": "turboquant-v3", "params": {"key_bits": 2, "value_bits": 2, "residual_window": 128}}),
        ("kivi_4",  {"type": "kivi", "params": {"bits": 4, "residual_window": 128}}),
        ("kivi_2",  {"type": "kivi", "params": {"bits": 2, "residual_window": 128}}),
    ]

    results = {}

    # Memory benchmarks
    print(f"\n=== Memory benchmark (seq_len={args.mem_seq_len}) ===")
    print(f"{'Method':>12s} | {'Peak MB':>10s} | {'Cache MB':>10s}")
    print("-" * 40)
    for name, method_config in methods:
        gc.collect(); torch.cuda.empty_cache()
        if method_config is None:
            factory = None
        else:
            mc = method_config
            factory = lambda mc=mc: create_cache(mc, info)
        try:
            mem = measure_memory(model, tokenizer, cache_factory=factory, seq_len=args.mem_seq_len)
            print(f"{name:>12s} | {mem['peak_memory_mb']:>10.1f} | {mem['cache_overhead_mb']:>10.1f}")
            results.setdefault(name, {})["memory"] = mem
        except Exception as e:
            print(f"{name:>12s} | ERROR: {str(e)[:60]}")
            results.setdefault(name, {})["memory"] = {"error": str(e)[:200]}

    # Latency benchmarks
    print(f"\n=== Latency benchmark (prompt={args.prompt_len}, gen={args.gen_len}) ===")
    print(f"{'Method':>12s} | {'Prefill(ms)':>12s} | {'Decode(ms)':>12s} | {'tok/s':>8s}")
    print("-" * 55)
    for name, method_config in methods:
        gc.collect(); torch.cuda.empty_cache()
        if method_config is None:
            factory = None
        else:
            mc = method_config
            factory = lambda mc=mc: create_cache(mc, info)
        try:
            lat = measure_latency(model, tokenizer, cache_factory=factory,
                                  prompt_len=args.prompt_len, gen_len=args.gen_len)
            print(f"{name:>12s} | {lat['prefill_ms']:>12.1f} | {lat['decode_ms']:>12.1f} | {lat['decode_tokens_per_sec']:>8.1f}")
            results.setdefault(name, {})["latency"] = lat
        except Exception as e:
            print(f"{name:>12s} | ERROR: {str(e)[:60]}")
            results.setdefault(name, {})["latency"] = {"error": str(e)[:200]}

    # Theoretical compression ratios
    print(f"\n=== Theoretical compression ratios (bits per KV element) ===")
    bits_per_elem = {"fp16": 16, "tq_k4v4": 4, "tq_k4v2": 3, "tq_k2v2": 2, "kivi_4": 4, "kivi_2": 2}
    for name, b in bits_per_elem.items():
        ratio = 16 / b
        print(f"  {name:>12s}: {b} bits → {ratio:.1f}x compression")
        results.setdefault(name, {})["theoretical_bits"] = b
        results[name]["theoretical_ratio"] = ratio

    # Save
    record = {
        "model": args.model,
        "model_info": info,
        "config": {"prompt_len": args.prompt_len, "gen_len": args.gen_len, "mem_seq_len": args.mem_seq_len},
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
