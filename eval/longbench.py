"""LongBench evaluation runner.

LongBench is a long-context benchmark with contexts in the 4K–32K range,
which is exactly where KV cache compression matters. We run 3 representative
sub-tasks:

- ``narrativeqa`` — Single-document question answering on books/movie scripts.
  Metric: F1.
- ``hotpotqa`` — Multi-document QA requiring multi-hop reasoning. Metric: F1.
- ``gov_report`` — Long-document summarization of government reports. Metric:
  ROUGE-L.

The dataset is loaded via ``datasets.load_dataset("THUDM/LongBench", <task>)``
(publicly available, no gating).
"""
import gc
import re
import string
from collections import Counter
from typing import Optional, Callable, List, Dict, Any

import torch

# Per-task max_new_tokens from LongBench's official config
TASK_GEN_LEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# Simple task-specific prompts (approximating LongBench's format)
TASK_PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
        "story as concisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
}


def _normalize_answer(s: str) -> str:
    """Lower, remove articles and punctuation, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _qa_metric(prediction: str, ground_truths: List[str]) -> float:
    """F1 over multiple ground truths — take the best."""
    return max(_f1_score(prediction, gt) for gt in ground_truths)


def _rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 (longest common subsequence)."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    m, n = len(pred_tokens), len(gt_tokens)
    # DP for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gt_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


TASK_METRIC = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
}


def _score_example(prediction: str, answers: List[str], metric: str) -> float:
    if metric == "f1":
        return _qa_metric(prediction, answers)
    elif metric == "rouge_l":
        return max(_rouge_l(prediction, gt) for gt in answers) if answers else 0.0
    else:
        # Fallback: exact match
        return float(any(_normalize_answer(prediction) == _normalize_answer(gt) for gt in answers))


def evaluate_longbench(
    model, tokenizer,
    cache_factory: Optional[Callable] = None,
    tasks: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    max_samples: int = 50,
    hook_compressor_config: Optional[Dict] = None,
    model_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run LongBench sub-tasks and return per-task metrics."""
    from datasets import load_dataset

    if tasks is None:
        tasks = ["narrativeqa", "hotpotqa", "gov_report"]

    use_hooks = hook_compressor_config is not None
    if use_hooks and model_info is None:
        raise ValueError("hook_compressor_config requires model_info")

    # Install hooks once around the whole evaluation (if hook mode)
    hook_ctx = None
    if use_hooks:
        from turboquant.hook_compression import HookCompressor
        hook_ctx = HookCompressor(model, model_info, hook_compressor_config)
        hook_ctx.__enter__()

    results_per_task: Dict[str, Dict[str, Any]] = {}

    try:
        for task in tasks:
            metric_name = TASK_METRIC.get(task, "f1")
            gen_len = TASK_GEN_LEN.get(task, max_new_tokens)
            gen_len = min(gen_len, max_new_tokens)
            prompt_tpl = TASK_PROMPT.get(task)
            if prompt_tpl is None:
                print(f"Skipping {task}: no prompt template")
                continue

            print(f"Loading LongBench[{task}]...")
            try:
                ds = load_dataset("THUDM/LongBench", task, split="test")
            except Exception as e:
                print(f"  Failed to load {task}: {e}")
                results_per_task[task] = {"error": str(e)[:200]}
                continue

            n = min(max_samples, len(ds))
            scores = []
            for i, ex in enumerate(ds.select(range(n))):
                ctx = ex.get("context", "") or ex.get("input", "")
                question = ex.get("input", "") if "context" in ex else ""
                answers = ex.get("answers", [])
                if isinstance(answers, str):
                    answers = [answers]
                if not answers:
                    continue

                prompt = prompt_tpl.format(context=ctx, input=question)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=30000)
                input_ids = inputs["input_ids"].to(model.device)
                attn_mask = inputs["attention_mask"].to(model.device)

                gen_kwargs = dict(
                    max_new_tokens=gen_len,
                    do_sample=False,
                    attention_mask=attn_mask,
                )
                if cache_factory is not None and not use_hooks:
                    gen_kwargs["past_key_values"] = cache_factory()
                    gen_kwargs["use_cache"] = True

                try:
                    with torch.no_grad():
                        out = model.generate(input_ids, **gen_kwargs)
                    new_tokens = out[0][input_ids.shape[1]:]
                    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    score = _score_example(pred, answers, metric_name)
                    scores.append(score)
                except Exception as e:
                    print(f"    sample {i} failed: {str(e)[:80]}")
                    continue

                gc.collect()
                torch.cuda.empty_cache()

            if scores:
                mean = sum(scores) / len(scores)
                results_per_task[task] = {
                    "score": mean,
                    "metric": metric_name,
                    "n_samples": len(scores),
                }
                print(f"  {task}: {metric_name}={mean:.4f} (n={len(scores)})")
            else:
                results_per_task[task] = {"error": "no samples scored"}

    finally:
        if hook_ctx is not None:
            hook_ctx.__exit__(None, None, None)

    return results_per_task
