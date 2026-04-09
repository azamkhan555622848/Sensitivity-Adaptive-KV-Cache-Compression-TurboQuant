"""Hardware metrics: latency and memory."""
import torch
import time
import gc
from typing import Callable, Optional

def measure_latency(model, tokenizer, cache_factory=None, prompt_len=2048, gen_len=128, warmup_runs=1):
    dummy_text = "Hello " * (prompt_len // 2)
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=prompt_len)
    input_ids = inputs["input_ids"].to(model.device)
    for _ in range(warmup_runs):
        cache = cache_factory() if cache_factory is not None else None
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=4, past_key_values=cache, use_cache=True, do_sample=False)
        gc.collect(); torch.cuda.empty_cache()
    cache = cache_factory() if cache_factory is not None else None
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    past = outputs.past_key_values
    decode_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len - 1):
            out = model(next_token, past_key_values=past, use_cache=True)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            past = out.past_key_values
    torch.cuda.synchronize()
    decode_end = time.perf_counter()
    prefill_ms = (prefill_end - prefill_start) * 1000
    decode_ms = (decode_end - decode_start) * 1000
    decode_tps = (gen_len - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms": round(decode_ms, 2),
            "decode_tokens_per_sec": round(decode_tps, 1), "total_ms": round(prefill_ms + decode_ms, 2)}

def measure_memory(model, tokenizer, cache_factory=None, seq_len=4096):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model_mem = torch.cuda.memory_allocated() / 1024 / 1024
    dummy_text = "Hello " * (seq_len // 2)
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = inputs["input_ids"].to(model.device)
    cache = cache_factory() if cache_factory is not None else None
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return {"peak_memory_mb": round(peak_mem, 1), "model_memory_mb": round(model_mem, 1),
            "cache_overhead_mb": round(peak_mem - model_mem, 1)}
