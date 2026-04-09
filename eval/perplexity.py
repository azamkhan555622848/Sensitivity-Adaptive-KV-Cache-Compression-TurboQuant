"""Perplexity evaluation on WikiText-2 and C4."""
import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable

def evaluate_perplexity_on_tokens(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
    return math.exp(loss.item())

def load_eval_tokens(tokenizer, dataset_name="wikitext2", max_tokens=0):
    from datasets import load_dataset
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        parts, n_tokens = [], 0
        for item in ds:
            parts.append(item["text"])
            n_tokens += len(item["text"].split()) * 1.3
            if max_tokens > 0 and n_tokens > max_tokens * 2:
                break
        text = "\n\n".join(parts)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens

def evaluate_perplexity(model, tokenizer, cache_factory=None, dataset_name="wikitext2",
                        max_seq_len=2048, stride=None, max_tokens=0, device="cuda"):
    if stride is None:
        stride = max_seq_len // 2
    tokens = load_eval_tokens(tokenizer, dataset_name, max_tokens)
    total_len = tokens.size(0)
    total_loss, total_count = 0.0, 0
    for begin in range(0, total_len - 1, stride):
        end = min(begin + max_seq_len, total_len)
        input_ids = tokens[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        if begin > 0:
            target_ids[:, :max_seq_len - stride] = -100
        cache = cache_factory() if cache_factory is not None else None
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache, use_cache=(cache is not None))
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        mask = shift_labels.view(-1) != -100
        if mask.any():
            total_loss += loss[mask].sum().item()
            total_count += mask.sum().item()
        if end >= total_len:
            break
    avg_loss = total_loss / total_count if total_count > 0 else float("inf")
    return {"perplexity": math.exp(avg_loss), "loss": avg_loss, "n_tokens": total_count}
