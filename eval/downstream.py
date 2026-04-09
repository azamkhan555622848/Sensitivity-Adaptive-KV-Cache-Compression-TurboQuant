"""Downstream task evaluation via lm-evaluation-harness."""
import torch
from typing import Callable, Optional, List, Dict

def evaluate_downstream_manual(model, tokenizer, cache_factory=None, tasks=None, max_samples=0):
    return {task: {"status": "manual_eval_not_yet_implemented"} for task in (tasks or ["mmlu"])}

def evaluate_downstream(model, tokenizer, cache_factory=None, tasks=None, num_fewshot=None):
    if tasks is None:
        tasks = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("WARNING: lm-eval not installed. Falling back to manual evaluation.")
        return evaluate_downstream_manual(model, tokenizer, cache_factory, tasks)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    if cache_factory is not None:
        original_generate = model.generate
        def patched_generate(*args, **kwargs):
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True
            return original_generate(*args, **kwargs)
        model.generate = patched_generate
    try:
        results = lm_eval.simple_evaluate(model=lm, tasks=tasks, batch_size=1)
    finally:
        if cache_factory is not None:
            model.generate = original_generate
    parsed = {}
    for task_name, task_result in results.get("results", {}).items():
        acc_key = None
        for k in ["acc,none", "acc_norm,none", "exact_match,strict-match"]:
            if k in task_result:
                acc_key = k
                break
        if acc_key:
            parsed[task_name] = {"accuracy": task_result[acc_key],
                                 "stderr": task_result.get(f"{acc_key.split(',')[0]}_stderr,none", 0)}
        else:
            parsed[task_name] = task_result
    return parsed
