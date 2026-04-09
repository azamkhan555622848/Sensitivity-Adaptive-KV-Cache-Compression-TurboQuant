#!/usr/bin/env python3
"""CLI for running TurboQuant experiments."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.runner import run_sweep

def main():
    parser = argparse.ArgumentParser(description="Run TurboQuant experiments")
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    parser.add_argument("--model", default=None, help="Filter to specific model")
    args = parser.parse_args()
    run_sweep(args.config, model_filter=args.model)

if __name__ == "__main__":
    main()
