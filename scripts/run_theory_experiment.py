#!/usr/bin/env python3
"""Synthetic controlled-GQA experiment for Section 5 (theory).

We take a single backbone model (Llama-3.1-8B) and vary the effective GQA
ratio by pooling/splitting KV heads, then measure WikiText-2 PPL under
TurboQuant K4V4 and KIVI-4 at each setting. The expected outcome from our
theory is that TurboQuant's relative degradation grows ~linearly in g while
KIVI's grows ~sqrt(g), so TQ's advantage shrinks as g increases.

We implement "pooling" the KV heads by writing a forward hook on k_proj /
v_proj that:
1. Splits the (B, S, H_kv * D) projection into H_kv heads.
2. Averages groups of (original_kv_heads / new_kv_heads) heads to reduce the
   number of distinct KV heads.
3. Broadcasts the pooled values back to the original H_kv slot positions.

This simulates a model with a different effective GQA ratio while keeping
all other weights and positions unchanged.

Note: this is a first-order approximation to varying GQA. A fully correct
experiment would retrain the attention layer, which is infeasible. The
synthetic experiment's purpose is to show the DIRECTIONAL prediction of our
theory, not to be a production-quality re-architecture.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from eval.perplexity import evaluate_perplexity
from turboquant.hook_compression import HookCompressor


class GQAPoolingHook:
    """Context manager that reduces effective KV heads by pooling.

    original: n_kv_heads_original
    target: n_kv_heads_target (must divide n_kv_heads_original)

    Installs forward hooks on k_proj/v_proj. Each hook reshapes the output
    into (B, S, n_kv_heads_original, D), averages groups of
    (n_kv_heads_original / n_kv_heads_target) consecutive heads, then
    broadcasts the pooled results back to fill all n_kv_heads_original slots
    (so downstream code doesn't see a shape change).
    """

    def __init__(self, model, model_info, n_kv_heads_target):
        self.model = model
        self.model_info = model_info
        self.n_kv_heads_original = model_info["n_kv_heads"]
        self.n_kv_heads_target = n_kv_heads_target
        if self.n_kv_heads_original % n_kv_heads_target != 0:
            raise ValueError(
                f"Target KV heads {n_kv_heads_target} must divide original "
                f"{self.n_kv_heads_original}"
            )
        self.group_size = self.n_kv_heads_original // n_kv_heads_target
        self.handles = []

    def __enter__(self):
        if self.group_size == 1:
            # No-op: target = original
            return self

        base = getattr(self.model, "model", self.model)
        layers = base.layers
        n_heads_orig = self.n_kv_heads_original
        head_dim = self.model_info["head_dim"]
        group_size = self.group_size

        def make_hook():
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return output
                B, S, D_total = out.shape
                if D_total != n_heads_orig * head_dim:
                    return output
                # (B, S, H_orig, D)
                x = out.view(B, S, n_heads_orig, head_dim)
                # Group-average: (B, S, H_target, group_size, D) -> mean over group_size
                pooled = x.view(B, S, n_heads_orig // group_size, group_size, head_dim).mean(dim=3)
                # Broadcast back to original shape by repeating
                expanded = pooled.unsqueeze(3).expand(-1, -1, -1, group_size, -1)
                expanded = expanded.reshape(B, S, n_heads_orig, head_dim).contiguous()
                x_out = expanded.view(B, S, D_total)
                if isinstance(output, tuple):
                    return (x_out,) + output[1:]
                return x_out
            return hook

        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            kp = getattr(attn, "k_proj", None)
            vp = getattr(attn, "v_proj", None)
            if kp is None or vp is None:
                continue
            self.handles.append(kp.register_forward_hook(make_hook()))
            self.handles.append(vp.register_forward_hook(make_hook()))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


def evaluate_at_gqa(model, tokenizer, model_info, n_kv_heads_target, method_config,
                     max_tokens=50000):
    """Run perplexity with a given (method, effective GQA) combination."""
    # The compression method_config is passed to HookCompressor too; we install
    # BOTH hooks — pooling first (outermost), then compression (innermost) so
    # compression sees the pooled tensors.
    with GQAPoolingHook(model, model_info, n_kv_heads_target):
        if method_config.get("type", "fp16") == "fp16":
            result = evaluate_perplexity(
                model, tokenizer, cache_factory=None,
                dataset_name="wikitext2", max_seq_len=2048,
                max_tokens=max_tokens,
            )
        else:
            # Use hook-mode compression. Note that the pooling hooks and
            # compression hooks are both forward hooks on the same modules;
            # PyTorch runs them in registration order, so pooling happens
            # FIRST (outer scope) and compression SECOND (inner scope).
            result = evaluate_perplexity(
                model, tokenizer, cache_factory=None,
                dataset_name="wikitext2", max_seq_len=2048,
                max_tokens=max_tokens,
                hook_compressor_config=method_config,
                model_info=model_info,
            )
    return result["perplexity"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                         help="Backbone model (must have n_kv_heads divisible "
                              "by all target values)")
    parser.add_argument("--gqa-targets", nargs="+", type=int,
                         default=[8, 4, 2, 1],  # effective n_kv_heads targets
                         help="Target n_kv_heads values. Llama-8B has 8 KV "
                              "heads originally; reducing to 4/2/1 increases "
                              "effective GQA ratio.")
    parser.add_argument("--output", default="results/theory/gqa_sweep.json")
    parser.add_argument("--max-tokens", type=int, default=30000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)
    model_info = get_model_info(model)
    n_heads_original = model_info["n_heads"]
    n_kv_original = model_info["n_kv_heads"]
    print(f"  Original: n_heads={n_heads_original}, n_kv_heads={n_kv_original}")

    methods = {
        "fp16": {"type": "fp16", "params": {}},
        "tq_k4v4": {"type": "turboquant-v3",
                    "params": {"key_bits": 4, "value_bits": 4, "residual_window": 128}},
        "kivi4": {"type": "kivi", "params": {"bits": 4, "residual_window": 128}},
    }

    results = []
    for target in args.gqa_targets:
        if n_kv_original % target != 0:
            print(f"Skipping target={target}: doesn't divide {n_kv_original}")
            continue
        effective_gqa = n_heads_original // target
        print(f"\n=== Target n_kv={target}, effective GQA={effective_gqa}:1 ===")
        row = {"n_kv_heads_target": target, "gqa_ratio": effective_gqa}
        for method_name, method_config in methods.items():
            t0 = time.time()
            try:
                ppl = evaluate_at_gqa(model, tokenizer, model_info, target,
                                       method_config, max_tokens=args.max_tokens)
            except Exception as e:
                print(f"  {method_name}: ERROR {str(e)[:80]}")
                row[method_name] = None
                continue
            elapsed = time.time() - t0
            print(f"  {method_name}: PPL={ppl:.4f}  ({elapsed:.0f}s)")
            row[method_name] = ppl
        results.append(row)

    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "n_heads_original": n_heads_original,
            "n_kv_heads_original": n_kv_original,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
