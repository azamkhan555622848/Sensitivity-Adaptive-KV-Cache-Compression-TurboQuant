#!/usr/bin/env python3
"""Profile outlier channels in KV cache and test outlier-aware compression."""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.model_loader import load_model, get_model_info
from turboquant.adaptive import calibrate
from turboquant.outlier import detect_outlier_channels, OutlierAwareMSECompressor
from turboquant.compressors_v3 import MSECompressor
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        os.makedirs("results/outlier", exist_ok=True)
        args.output = f"results/outlier/{safe_name}_outlier_profile.json"

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    info = get_model_info(model)
    print(f"  n_layers={info['n_layers']}, head_dim={info['head_dim']}, n_kv_heads={info['n_kv_heads']}")

    print(f"\nCalibrating ({args.n_samples} samples)...")
    kv_data = calibrate(model, tokenizer, n_samples=args.n_samples, max_seq_len=args.max_seq_len)

    print(f"\nOutlier channel analysis (threshold={args.threshold}):")
    print(f"{'Layer':>6} | {'K outliers':>10} | {'V outliers':>10} | {'K channels':>15} | {'V channels':>15}")
    print("-" * 75)

    layer_outliers = {}
    for layer_idx in sorted(kv_data.keys()):
        k_data = kv_data[layer_idx]["keys"]
        v_data = kv_data[layer_idx]["values"]

        k_mask = detect_outlier_channels(k_data, threshold_factor=args.threshold)
        v_mask = detect_outlier_channels(v_data, threshold_factor=args.threshold)

        k_outlier_idx = k_mask.nonzero(as_tuple=True)[0].tolist()
        v_outlier_idx = v_mask.nonzero(as_tuple=True)[0].tolist()

        print(f"{layer_idx:>6} | {k_mask.sum().item():>10} | {v_mask.sum().item():>10} | "
              f"{str(k_outlier_idx[:5]):>15} | {str(v_outlier_idx[:5]):>15}")

        layer_outliers[layer_idx] = {
            "key_outlier_count": k_mask.sum().item(),
            "value_outlier_count": v_mask.sum().item(),
            "key_outlier_channels": k_outlier_idx,
            "value_outlier_channels": v_outlier_idx,
        }

    # Compare compression quality: standard vs outlier-aware
    print(f"\nCompression quality comparison (4-bit, layer 0):")
    k0 = kv_data[0]["keys"]
    v0 = kv_data[0]["values"]
    head_dim = info["head_dim"]

    # Standard MSE
    comp_std = MSECompressor(head_dim, bits=4, seed=42, device="cpu")
    ck = comp_std.compress(k0)
    dk = comp_std.decompress(ck)
    mse_std_k = (k0 - dk).pow(2).mean().item()

    cv = comp_std.compress(v0)
    dv = comp_std.decompress(cv)
    mse_std_v = (v0 - dv).pow(2).mean().item()

    # Outlier-aware
    k_mask = detect_outlier_channels(k0, threshold_factor=args.threshold)
    v_mask = detect_outlier_channels(v0, threshold_factor=args.threshold)

    if k_mask.sum() > 0:
        comp_out_k = OutlierAwareMSECompressor(head_dim, bits=4, outlier_mask=k_mask, seed=42, device="cpu")
        ck_out = comp_out_k.compress(k0)
        dk_out = comp_out_k.decompress(ck_out)
        mse_out_k = (k0 - dk_out).pow(2).mean().item()
    else:
        mse_out_k = mse_std_k

    if v_mask.sum() > 0:
        comp_out_v = OutlierAwareMSECompressor(head_dim, bits=4, outlier_mask=v_mask, seed=42, device="cpu")
        cv_out = comp_out_v.compress(v0)
        dv_out = comp_out_v.decompress(cv_out)
        mse_out_v = (v0 - dv_out).pow(2).mean().item()
    else:
        mse_out_v = mse_std_v

    print(f"  Standard K MSE: {mse_std_k:.6f}")
    print(f"  Outlier  K MSE: {mse_out_k:.6f} ({k_mask.sum().item()} outlier channels)")
    print(f"  Standard V MSE: {mse_std_v:.6f}")
    print(f"  Outlier  V MSE: {mse_out_v:.6f} ({v_mask.sum().item()} outlier channels)")
    if mse_std_k > 0:
        print(f"  K improvement:  {(1 - mse_out_k/mse_std_k)*100:.1f}%")
    if mse_std_v > 0:
        print(f"  V improvement:  {(1 - mse_out_v/mse_std_v)*100:.1f}%")

    # Save
    result = {
        "model": args.model,
        "model_info": info,
        "threshold": args.threshold,
        "n_samples": args.n_samples,
        "layer_outliers": {str(k): v for k, v in layer_outliers.items()},
        "layer0_comparison": {
            "standard_key_mse": mse_std_k, "outlier_key_mse": mse_out_k,
            "standard_value_mse": mse_std_v, "outlier_value_mse": mse_out_v,
            "key_outlier_count": k_mask.sum().item(),
            "value_outlier_count": v_mask.sum().item(),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
