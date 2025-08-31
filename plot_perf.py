#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# load & light cleaning
# ---------------------------
csv = "perf_results.csv"
df = pd.read_csv(csv)

# keep only rows that actually ran
df = df.dropna(subset=["total"])

# nicer types
for col in ["batches","nsub","arows","streams","chunk_cap","write","one_based"]:
    if col in df: df[col] = df[col].astype(int)

# helper for safe division
def safe_div(a, b): 
    return np.where(np.abs(b) > 1e-9, a / b, np.nan)

# ---------------------------
# basic plots (saved as PNGs)
# ---------------------------
outdir = Path("perf_plots"); outdir.mkdir(exist_ok=True)

# 1) Total time vs nsub, colored by streams, one line per batches
for B in sorted(df["batches"].unique()):
    d = df[df["batches"] == B]
    fig = plt.figure(figsize=(6,4))
    for S in sorted(d["streams"].unique()):
        dd = d[d["streams"]==S].groupby("nsub", as_index=False)["total"].median().sort_values("nsub")
        plt.plot(dd["nsub"], dd["total"], marker="o", label=f"streams={S}")
    plt.title(f"Total time vs nsub (batches={B})")
    plt.xlabel("nsub (B size)")
    plt.ylabel("Total time (ms)")
    plt.grid(True, alpha=0.3); plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"total_vs_nsub_batches{B}.png", dpi=160)
    plt.close(fig)

# 2) SpMM time vs A rows, split by ALG
for ALG in sorted(df["alg"].dropna().unique()):
    d = df[df["alg"] == ALG]
    fig = plt.figure(figsize=(6,4))
    dd = d.groupby("arows", as_index=False)["spmm"].median().sort_values("arows")
    plt.plot(dd["arows"], dd["spmm"], marker="o")
    plt.title(f"SpMM time vs A rows (ALG={ALG})")
    plt.xlabel("A rows per batch")
    plt.ylabel("SpMM total (ms)")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"spmm_vs_arows_alg_{ALG}.png", dpi=160)
    plt.close(fig)

# 3) Algorithm comparison bar (median total), grouped by ALG
fig = plt.figure(figsize=(6,4))
dd = df.groupby("alg", as_index=False)["total"].median().sort_values("total")
plt.bar(dd["alg"], dd["total"])
plt.title("Median total time by SpMM algorithm")
plt.ylabel("Total time (ms)")
for x,y in zip(dd["alg"], dd["total"]):
    plt.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
plt.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(outdir / "alg_comparison_total.png", dpi=160)
plt.close(fig)

# 4) Streams scaling (median total) for each nsub
fig = plt.figure(figsize=(7,4))
for K in sorted(df["nsub"].unique()):
    d = df[df["nsub"]==K].groupby("streams", as_index=False)["total"].median().sort_values("streams")
    plt.plot(d["streams"], d["total"], marker="o", label=f"nsub={K}")
plt.title("Streams scaling (lower is better)")
plt.xlabel("# streams"); plt.ylabel("Total time (ms)")
plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(outdir / "streams_scaling.png", dpi=160)
plt.close(fig)

# 5) I/O effect (write on/off), median totals for each nsub
fig = plt.figure(figsize=(7,4))
for wflag in sorted(df["write"].unique()):
    d = df[df["write"]==wflag].groupby("nsub", as_index=False)["total"].median().sort_values("nsub")
    plt.plot(d["nsub"], d["total"], marker="o", label=f"write={wflag}")
plt.title("Effect of writing T/D binaries")
plt.xlabel("nsub"); plt.ylabel("Total time (ms)")
plt.grid(True, alpha=0.3); plt.legend()
fig.tight_layout()
fig.savefig(outdir / "write_effect.png", dpi=160)
plt.close(fig)

# 6) (Optional) very rough GFLOP/s estimate for SpMM
#    Using density as a proxy (not exact): Flops ≈ 2*nnz*A_cols for T + 2*nnz*A_rows for D
if {"density","arows","nsub","spmm"}.issubset(df.columns):
    # Approx nnz per A batch ~ density * arows * nsub  (proxy!)
    nnz_est = df["density"] * df["arows"] * df["nsub"]
    flops_est = 2*nnz_est*df["nsub"] + 2*nnz_est*df["arows"]
    gflops = safe_div(flops_est, df["spmm"]/1000.0) / 1e9

    fig = plt.figure(figsize=(6,4))
    # average by nsub
    d = pd.DataFrame({"nsub":df["nsub"], "gflops":gflops}).groupby("nsub", as_index=False)["gflops"].median()
    plt.plot(d["nsub"], d["gflops"], marker="o")
    plt.title("Approx SpMM GFLOP/s vs nsub (proxy)")
    plt.xlabel("nsub"); plt.ylabel("GFLOP/s (approx)")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "approx_gflops_vs_nsub.png", dpi=160)
    plt.close(fig)

print("Saved plots to", outdir)
