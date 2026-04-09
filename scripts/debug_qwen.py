#!/usr/bin/env python3
"""Debug Qwen compression quality."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.model_loader import load_model, get_model_info
from eval.perplexity import evaluate_perplexity
from turboquant.baselines.registry import create_cache

print("Loading Qwen2.5-3B...")
model, tokenizer = load_model("Qwen/Qwen2.5-3B-Instruct")
info = get_model_info(model)
gqa_ratio = info["n_heads"] // info["n_kv_heads"]
print(f"n_kv_heads={info['n_kv_heads']} (GQA ratio={gqa_ratio}:1)")

configs = [
    ("FP16", None),
    ("TQ K4V4 rw=128",  {"type": "turboquant-v3", "params": {"key_bits": 4, "value_bits": 4, "residual_window": 128}}),
    ("TQ K4V4 rw=512",  {"type": "turboquant-v3", "params": {"key_bits": 4, "value_bits": 4, "residual_window": 512}}),
    ("TQ K4V4 rw=1024", {"type": "turboquant-v3", "params": {"key_bits": 4, "value_bits": 4, "residual_window": 1024}}),
    ("TQ K6V4 rw=128",  {"type": "turboquant-v3", "params": {"key_bits": 6, "value_bits": 4, "residual_window": 128}}),
    ("TQ K8V8 rw=128",  {"type": "turboquant-v3", "params": {"key_bits": 8, "value_bits": 8, "residual_window": 128}}),
]

for name, method_config in configs:
    if method_config is None:
        factory = None
    else:
        mc = method_config
        factory = lambda: create_cache(mc, info)
    result = evaluate_perplexity(model, tokenizer, cache_factory=factory,
                                 dataset_name="wikitext2", max_seq_len=2048, max_tokens=50000)
    print(f"  {name}: PPL={result['perplexity']:.4f} ({result['n_tokens']} tokens)")
