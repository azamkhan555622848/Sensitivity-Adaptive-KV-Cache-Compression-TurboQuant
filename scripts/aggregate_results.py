#!/usr/bin/env python3
"""Aggregate experiment JSON results into LaTeX tables and CSV."""
import argparse, json, os, sys
import pandas as pd

def load_results(results_dir):
    records = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fp:
                records.append(json.load(fp))
    return records

def make_perplexity_table(records):
    rows = []
    for r in records:
        if r["benchmark"] != "perplexity":
            continue
        model = r["model"].split("/")[-1]
        method = r["method"]
        for dataset, metrics in r["results"].items():
            if isinstance(metrics, dict) and "perplexity" in metrics:
                rows.append({"Model": model, "Method": method, "Dataset": dataset, "PPL": round(metrics["perplexity"], 2)})
    return pd.DataFrame(rows)

def to_latex(df, caption=""):
    return df.to_latex(index=False, caption=caption, label="tab:results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", default="tables/")
    args = parser.parse_args()
    records = load_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}"); sys.exit(1)
    os.makedirs(args.output, exist_ok=True)
    ppl_df = make_perplexity_table(records)
    if not ppl_df.empty:
        ppl_df.to_csv(os.path.join(args.output, "perplexity.csv"), index=False)
        with open(os.path.join(args.output, "perplexity.tex"), "w") as f:
            f.write(to_latex(ppl_df, "Perplexity Results"))
        print(f"Perplexity table: {len(ppl_df)} rows")
        print(ppl_df.to_string(index=False))
    print(f"\nTotal results loaded: {len(records)}")

if __name__ == "__main__":
    main()
