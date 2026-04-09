"""Experiment runner: config -> model -> cache -> benchmarks -> JSON."""
import json, os, time, subprocess
import torch
import gc
from typing import Optional
from .config import load_sweep
from .model_loader import load_model, get_model_info
from .perplexity import evaluate_perplexity
from .needle import evaluate_needle
from turboquant.baselines.registry import create_cache

def _get_git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except: return "unknown"

def _make_cache_factory(method_config, model_info):
    def factory():
        return create_cache(method_config, model_info)
    return factory

def run_single(model, tokenizer, model_info, model_name, method_config, benchmark_config, output_dir):
    method_dict = {"type": method_config.type, "params": method_config.params}
    cache_factory = _make_cache_factory(method_dict, model_info)
    if benchmark_config.type == "perplexity":
        params = benchmark_config.params
        results = {}
        for ds in params.get("datasets", ["wikitext2"]):
            results[ds] = evaluate_perplexity(model, tokenizer, cache_factory, dataset_name=ds,
                                               max_seq_len=params.get("max_seq_len", 2048))
    elif benchmark_config.type == "needle":
        params = benchmark_config.params
        raw = evaluate_needle(model, tokenizer, cache_factory,
                              context_lengths=params.get("context_lengths"),
                              needle_positions=params.get("positions"))
        results = {f"{k[0]}_{k[1]}": v for k, v in raw.items()}
    elif benchmark_config.type == "downstream":
        from .downstream import evaluate_downstream
        results = evaluate_downstream(model, tokenizer, cache_factory,
                                       tasks=benchmark_config.params.get("tasks"))
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_config.type}")
    record = {"model": model_name, "method": method_config.type,
              "method_config": method_config.params, "benchmark": benchmark_config.type,
              "benchmark_config": benchmark_config.params, "results": results,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": _get_git_sha()}
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.split('/')[-1]}_{method_config.type}_{benchmark_config.type}_{int(time.time())}.json"
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(record, f, indent=2, default=str)
    return record

def run_sweep(config_path, model_filter=None):
    config = load_sweep(config_path)
    for model_cfg in config.models:
        if model_filter and model_filter not in model_cfg.name:
            continue
        print(f"\n{'='*60}\nLoading model: {model_cfg.name}\n{'='*60}")
        model, tokenizer = load_model(model_cfg.name, dtype=model_cfg.dtype)
        model_info = get_model_info(model)
        for method_cfg in config.methods:
            for bench_cfg in config.benchmarks:
                print(f"\n  Method: {method_cfg.type} | Benchmark: {bench_cfg.type}")
                try:
                    record = run_single(model, tokenizer, model_info, model_cfg.name,
                                       method_cfg, bench_cfg, config.output_dir)
                    print(f"    Done: {record.get('results', {})}")
                except Exception as e:
                    print(f"    ERROR: {e}")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
