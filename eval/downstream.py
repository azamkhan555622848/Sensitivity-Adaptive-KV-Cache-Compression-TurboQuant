"""Downstream task evaluation via lm-evaluation-harness.

The key challenge addressed here: lm-eval's loglikelihood tasks (MMLU,
ARC, HellaSwag, WinoGrande) call ``model.forward`` (via HFLM._model_call)
for every example, while generation tasks (GSM8K) call ``model.generate``.
To evaluate a compressed KV cache on both kinds of tasks we must intercept
both paths.

We subclass ``HFLM`` as ``CompressedHFLM`` that:
1. Overrides ``_model_call`` to inject a fresh compressed cache on every
   forward pass for loglikelihood evaluation.
2. Wraps ``_model_generate`` to inject a fresh compressed cache on every
   generation call.

For Nemotron-style models (incompatible cache), we fall back to the
hook-based compression path — install forward hooks on k_proj/v_proj at
task start, remove at task end.
"""
import torch
from typing import Callable, Optional, List, Dict, Any


def evaluate_downstream_manual(model, tokenizer, cache_factory=None, tasks=None, max_samples=0):
    return {task: {"status": "manual_eval_not_yet_implemented"} for task in (tasks or ["mmlu"])}


def evaluate_downstream(model, tokenizer, cache_factory=None, tasks=None,
                        num_fewshot=None, limit=None, batch_size=1,
                        hook_compressor_config=None, model_info=None):
    """Run lm-evaluation-harness tasks with KV cache compression.

    Parameters
    ----------
    model, tokenizer : HuggingFace model and tokenizer
    cache_factory : callable returning a fresh DynamicCache subclass, or None.
        Used in cache-mode (most models).
    tasks : list of task names for lm-eval.
    num_fewshot : override few-shot count (None = use task default).
    limit : int, optional. Limits number of examples per task (for smoke tests).
    batch_size : lm-eval batch size.
    hook_compressor_config : dict, optional. If set, uses hook-based
        compression instead of cache_factory. Required for Nemotron-style
        models.
    model_info : dict. Required if hook_compressor_config is set.
    """
    if tasks is None:
        tasks = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("WARNING: lm-eval not installed. Falling back to manual evaluation.")
        return evaluate_downstream_manual(model, tokenizer, cache_factory, tasks)

    use_hooks = hook_compressor_config is not None
    if use_hooks and model_info is None:
        raise ValueError("hook_compressor_config requires model_info to be passed")

    # Build a CompressedHFLM if we have a cache_factory. Otherwise return plain HFLM
    # (which will be FP16 by default, or compression will be injected via hooks).
    class CompressedHFLM(HFLM):
        """HFLM subclass that injects a compressed KV cache on every forward."""

        def __init__(self, *args, cache_factory=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache_factory = cache_factory

        def _model_call(self, inps, attn_mask=None, labels=None):
            with torch.no_grad():
                if self._cache_factory is None:
                    return self.model(inps).logits
                cache = self._cache_factory()
                return self.model(
                    inps,
                    past_key_values=cache,
                    use_cache=True,
                ).logits

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            # Inject a fresh cache for every generation call.
            if self._cache_factory is not None:
                cache = self._cache_factory()
                generation_kwargs["past_key_values"] = cache
                generation_kwargs["use_cache"] = True
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

    if use_hooks:
        # Hook mode: install hooks once around the entire evaluation.
        from turboquant.hook_compression import HookCompressor
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        hook_ctx = HookCompressor(model, model_info, hook_compressor_config)
        hook_ctx.__enter__()
        try:
            kw = {"model": lm, "tasks": tasks, "batch_size": batch_size}
            if num_fewshot is not None:
                kw["num_fewshot"] = num_fewshot
            if limit is not None:
                kw["limit"] = limit
            results = lm_eval.simple_evaluate(**kw)
        finally:
            hook_ctx.__exit__(None, None, None)
    else:
        lm = CompressedHFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
            cache_factory=cache_factory,
        )
        kw = {"model": lm, "tasks": tasks, "batch_size": batch_size}
        if num_fewshot is not None:
            kw["num_fewshot"] = num_fewshot
        if limit is not None:
            kw["limit"] = limit
        results = lm_eval.simple_evaluate(**kw)

    parsed = {}
    for task_name, task_result in results.get("results", {}).items():
        acc_key = None
        for k in ["acc,none", "acc_norm,none", "exact_match,strict-match",
                  "exact_match,flexible-extract"]:
            if k in task_result:
                acc_key = k
                break
        if acc_key:
            stderr_key = f"{acc_key.split(',')[0]}_stderr,none"
            parsed[task_name] = {
                "accuracy": task_result[acc_key],
                "stderr": task_result.get(stderr_key, 0),
                "metric": acc_key,
            }
        else:
            parsed[task_name] = task_result
    return parsed
