#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd


def clip_positive(x, min_val=1e-3):
    return np.clip(x, a_min=min_val, a_max=None)


def gen_bw_csv(n: int, mean: float, std: float, col_name: str):
    """
    Generate a toy bandwidth CSV with approximately normal distribution.
    Values are clipped to be positive.
    """
    x = np.random.normal(loc=mean, scale=std, size=n)
    x = clip_positive(x, min_val=max(1e-3, mean * 0.01))
    return pd.DataFrame({col_name: x})


def gen_device_csv(n: int):
    """
    Generate a toy "Headset device performance.csv" compatible with moq_client_dispatch.py.
    Required columns:
      - GPU Clock (MHz)
      - RAM(GB)
      - Refresh Rate (Hz)
      - Resolution (per eye) (must contain the '×' delimiter)
    """
    rng = np.random.default_rng(42)

    gpu = rng.normal(loc=900, scale=180, size=n)
    gpu = clip_positive(gpu, min_val=200)

    ram = rng.normal(loc=8.0, scale=1.8, size=n)
    ram = clip_positive(ram, min_val=2.0)

    refresh = rng.normal(loc=90, scale=15, size=n)
    refresh = clip_positive(refresh, min_val=60)

    # Must contain the '×' character because the client parses:
    # int(str(x).split('×')[0])
    resolutions = ["1920×1080", "2560×1440", "3840×2160"]
    res = rng.choice(resolutions, size=n, replace=True)

    return pd.DataFrame(
        {
            "GPU Clock (MHz)": gpu,
            "RAM(GB)": ram,
            "Refresh Rate (Hz)": refresh,
            "Resolution (per eye)": res,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Generate lightweight toy datasets for Review Artifact.")
    parser.add_argument("--out_dir", type=str, default="datasets", help="Output directory (default: datasets/)")
    parser.add_argument("--rows", type=int, default=100, help="Number of rows per toy bandwidth CSV (default: 100)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n = args.rows

    # Bandwidth toy datasets
    # - wifi_clean.csv uses "bytes_sec (Mbps)"
    df_wifi = gen_bw_csv(n=n, mean=20.0, std=6.0, col_name="bytes_sec (Mbps)")
    df_wifi.to_csv(os.path.join(out_dir, "wifi_clean.csv"), index=False)

    # - 4G-network-data_clean.csv uses "DL_bitrate_Mbps"
    df_4g = gen_bw_csv(n=n, mean=12.0, std=3.0, col_name="DL_bitrate_Mbps")
    df_4g.to_csv(os.path.join(out_dir, "4G-network-data_clean.csv"), index=False)

    # - 5g_final_trace.csv uses "bytes_sec (Mbps)"
    df_5g = gen_bw_csv(n=n, mean=60.0, std=18.0, col_name="bytes_sec (Mbps)")
    df_5g.to_csv(os.path.join(out_dir, "5g_final_trace.csv"), index=False)

    # - Optic_Bandwidth_clean_2.csv uses "bytes_sec (Mbps)"
    df_fiber = gen_bw_csv(n=n, mean=250.0, std=80.0, col_name="bytes_sec (Mbps)")
    df_fiber.to_csv(os.path.join(out_dir, "Optic_Bandwidth_clean_2.csv"), index=False)

    # Device performance toy dataset
    df_dev = gen_device_csv(n=max(20, n))
    df_dev.to_csv(os.path.join(out_dir, "Headset device performance.csv"), index=False)

    print(f"[generate_toy_data.py] Wrote toy datasets to: {out_dir}")
    print("[generate_toy_data.py] Files:")
    for fn in [
        "wifi_clean.csv",
        "4G-network-data_clean.csv",
        "5g_final_trace.csv",
        "Optic_Bandwidth_clean_2.csv",
        "Headset device performance.csv",
    ]:
        print(f"  - {os.path.join(out_dir, fn)}")


if __name__ == "__main__":
    main()

