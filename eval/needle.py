"""Needle-in-haystack evaluation across context lengths and positions."""
import torch
import gc
from typing import Callable, Optional, List

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED_EXACT = "AURORA-7749"
EXPECTED_PARTIAL = ["AURORA", "7749"]
FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""

def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.5):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return (f"Read the following document carefully:\n\n{haystack}\n\n"
            f"What is the secret project code name mentioned in the document? "
            f"Answer with just the code name, nothing else.")

def classify_response(response):
    resp_lower = response.lower()
    if EXPECTED_EXACT.lower() in resp_lower:
        return "EXACT"
    if all(p.lower() in resp_lower for p in EXPECTED_PARTIAL):
        return "PARTIAL"
    return "MISS"

def evaluate_needle(model, tokenizer, cache_factory=None, context_lengths=None,
                    needle_positions=None, max_new_tokens=32):
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384, 32768]
    if needle_positions is None:
        needle_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    for ctx_len in context_lengths:
        for pos in needle_positions:
            prompt = build_prompt(tokenizer, ctx_len, pos)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len + 512)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            cache = cache_factory() if cache_factory is not None else None
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = model.generate(input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    past_key_values=cache, use_cache=True)
            new_tokens = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results[(ctx_len, pos)] = {"result": classify_response(response),
                                       "response": response[:100], "n_input_tokens": input_ids.shape[1]}
            gc.collect()
            torch.cuda.empty_cache()
    return results
