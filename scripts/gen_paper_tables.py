#!/usr/bin/env python3
"""Generate LaTeX tables from experiment results for the paper."""
import json, glob, os, sys
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
os.makedirs(os.path.join(PAPER_DIR, "tables"), exist_ok=True)

def load_all_results():
    """Load all experiment results into a structured dict."""
    all_results = []
    for sweep_dir in glob.glob(os.path.join(RESULTS_DIR, "*")):
        if not os.path.isdir(sweep_dir):
            continue
        if os.path.basename(sweep_dir) in ("benchmarks", "calibration", "outlier"):
            continue
        for f in glob.glob(os.path.join(sweep_dir, "*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                all_results.append((os.path.basename(sweep_dir), data))
            except Exception as e:
                print(f"Skipped {f}: {e}", file=sys.stderr)
    return all_results

def get_ppl(rec, dataset="wikitext2"):
    try:
        return rec["results"][dataset]["perplexity"]
    except (KeyError, TypeError):
        return None

def format_method(method, config):
    if method == "fp16":
        return "FP16"
    if method == "kivi":
        bits = config.get("bits", 4)
        return f"KIVI-{bits}"
    if method == "turboquant-v3":
        kb = config.get("key_bits", 4)
        vb = config.get("value_bits", 4)
        rw = config.get("residual_window", 128)
        if rw == 128:
            return f"TQ K{kb}V{vb}"
        return f"TQ K{kb}V{vb} (rw={rw})"
    if method == "turboquant-adaptive":
        budget = config.get("budget", 4.0)
        return f"TQ-Adaptive ({budget:.1f})"
    if method == "turboquant-outlier":
        kb = config.get("key_bits", 4)
        vb = config.get("value_bits", 4)
        return f"TQ-Outlier K{kb}V{vb}"
    if method == "turboquant-adaptive-outlier":
        kb = config.get("key_bits", 4)
        vb = config.get("value_bits", 4)
        return f"TQ-Adapt+Outlier K{kb}V{vb}"
    return method

def table_main_results():
    """Main PPL table: models x methods."""
    results = load_all_results()
    # Collect FP16 baselines and key 4-bit results
    by_model = defaultdict(dict)
    for sweep, rec in results:
        if rec.get("benchmark") != "perplexity":
            continue
        model = rec["model"].split("/")[-1]
        ppl = get_ppl(rec)
        if ppl is None:
            continue
        method_str = format_method(rec["method"], rec["method_config"])
        # Deduplicate: keep first (most recent usually)
        if method_str not in by_model[model]:
            by_model[model][method_str] = ppl

    # Models of interest
    model_order = [
        ("gemma-3-27b-it", "Gemma-3-27B", "2:1"),
        ("Mistral-7B-Instruct-v0.3", "Mistral-7B", "4:1"),
        ("Llama-3.1-8B-Instruct", "Llama-3.1-8B", "4:1"),
        ("Qwen2.5-32B-Instruct", "Qwen2.5-32B", "5:1"),
        ("Qwen2.5-3B-Instruct", "Qwen2.5-3B", "8:1"),
    ]
    method_cols = ["FP16", "TQ K4V4", "TQ K4V2", "TQ K2V2", "KIVI-4", "KIVI-2"]

    lines = []
    lines.append("\\begin{tabular}{l c c c c c c c}")
    lines.append("\\toprule")
    lines.append("Model & GQA & " + " & ".join(method_cols) + " \\\\")
    lines.append("\\midrule")
    for model_key, display, gqa in model_order:
        row = [display, gqa]
        fp16 = by_model.get(model_key, {}).get("FP16")
        for method in method_cols:
            val = by_model.get(model_key, {}).get(method)
            if val is None:
                row.append("--")
            elif val > 1000:
                row.append(f"\\textbf{{collapse}}")
            else:
                cell = f"{val:.2f}"
                if method == "FP16" or fp16 is None:
                    row.append(cell)
                else:
                    delta = (val - fp16) / fp16 * 100
                    row.append(f"{cell} {{\\small(+{delta:.1f}\\%)}}")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(os.path.join(PAPER_DIR, "tables", "main_ppl.tex"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote main_ppl.tex")
    print("\n".join(lines))
    return by_model

def table_qwen_adaptive():
    """Qwen budget sweep + outlier results."""
    results = load_all_results()
    rows = []
    for sweep, rec in results:
        if "Qwen" not in rec.get("model", ""):
            continue
        if rec.get("benchmark") != "perplexity":
            continue
        ppl = get_ppl(rec)
        if ppl is None:
            continue
        method = rec["method"]
        mc = rec["method_config"]
        if method == "fp16":
            label, bits = "FP16", 16.0
        elif method == "kivi":
            b = mc.get("bits", 4)
            label, bits = f"KIVI-{b}", float(b)
        elif method == "turboquant-v3":
            kb = mc.get("key_bits", 4); vb = mc.get("value_bits", 4); rw = mc.get("residual_window", 128)
            if rw != 128:
                continue
            label = f"TQ K{kb}V{vb} (uniform)"
            bits = (kb + vb) / 2
        elif method == "turboquant-adaptive":
            budget = mc.get("budget", 4.0)
            label = f"TQ-Adaptive budget={budget}"
            bits = budget
        elif method == "turboquant-outlier":
            kb = mc.get("key_bits", 4); vb = mc.get("value_bits", 4)
            label = f"TQ-Outlier K{kb}V{vb}"
            bits = (kb + vb) / 2 + 0.3  # outlier overhead
        elif method == "turboquant-adaptive-outlier":
            label = "TQ-Adapt+Outlier K4V4"
            bits = 4.3
        else:
            continue
        rows.append((label, bits, ppl))

    # Dedupe
    seen = {}
    for l, b, p in rows:
        if l not in seen:
            seen[l] = (b, p)
    rows = [(l, b, p) for l, (b, p) in seen.items()]
    rows.sort(key=lambda x: (x[1], x[2]))

    lines = []
    lines.append("\\begin{tabular}{l c c c}")
    lines.append("\\toprule")
    lines.append("Method & Avg bits & PPL & $\\Delta$ FP16 \\\\")
    lines.append("\\midrule")
    fp16_ppl = next((p for l, b, p in rows if l == "FP16"), None)
    for label, bits, ppl in rows:
        if ppl > 1000:
            cell = "\\textbf{collapse}"
        else:
            cell = f"{ppl:.2f}"
        delta = ""
        if fp16_ppl and label != "FP16":
            if ppl > 1000:
                delta = "--"
            else:
                delta = f"+{(ppl - fp16_ppl) / fp16_ppl * 100:.1f}\\%"
        lines.append(f"{label} & {bits:.1f} & {cell} & {delta} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(os.path.join(PAPER_DIR, "tables", "qwen_adaptive.tex"), "w") as f:
        f.write("\n".join(lines))
    print("\nWrote qwen_adaptive.tex")
    print("\n".join(lines))

def table_hardware():
    """Hardware benchmark table."""
    benchmarks = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "benchmarks", "*.json")):
        with open(f) as fh:
            data = json.load(fh)
        model = data["model"].split("/")[-1]
        benchmarks[model] = data["results"]

    model_order = [
        ("gemma-3-27b-it", "Gemma-3-27B"),
        ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
        ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
        ("Qwen2.5-32B-Instruct", "Qwen2.5-32B"),
        ("Qwen2.5-3B-Instruct", "Qwen2.5-3B"),
    ]
    methods = [("fp16", "FP16"), ("tq_k4v4", "TQ K4V4"), ("kivi_4", "KIVI-4")]

    lines = []
    lines.append("\\begin{tabular}{l l c c c}")
    lines.append("\\toprule")
    lines.append("Model & Method & Prefill (ms) & Decode tok/s & KV Mem (MB) \\\\")
    lines.append("\\midrule")
    for model_key, display in model_order:
        res = benchmarks.get(model_key, {})
        if not res:
            continue
        for method_key, method_name in methods:
            m = res.get(method_key, {})
            lat = m.get("latency", {})
            mem = m.get("memory", {})
            prefill = lat.get("prefill_ms", "--")
            tps = lat.get("decode_tokens_per_sec", "--")
            cache_mem = mem.get("cache_overhead_mb", "--")
            prefill_s = f"{prefill:.0f}" if isinstance(prefill, (int, float)) else prefill
            tps_s = f"{tps:.1f}" if isinstance(tps, (int, float)) else tps
            mem_s = f"{cache_mem:.0f}" if isinstance(cache_mem, (int, float)) else cache_mem
            row = f"{display if method_key == 'fp16' else ''} & {method_name} & {prefill_s} & {tps_s} & {mem_s} \\\\"
            lines.append(row)
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(os.path.join(PAPER_DIR, "tables", "hardware.tex"), "w") as f:
        f.write("\n".join(lines))
    print("\nWrote hardware.tex")
    print("\n".join(lines))

if __name__ == "__main__":
    table_main_results()
    table_qwen_adaptive()
    table_hardware()
