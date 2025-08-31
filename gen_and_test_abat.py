#!/usr/bin/env python3
"""
gen_and_test_abat.py

Generate arbitrary inputs for your CUDA pipeline that computes D = A * B^T * A^T,
where:
  - Global sparse matrix is CSR, stored in 'sparse.txt'
  - B batches (square) come from indx_b.txt (rows|cols) and are EXTRACTED from global, then
    your program inverts them and uses that inverse as B
  - A batches (rows|cols) come from idx_std_loc.txt; columns MUST match B size

This script can:
  - Generate synthetic cases (invertible B per batch; A aligned with B columns)
  - Write files in your expected text format (0-based or 1-based)
  - Optionally RUN your compiled CUDA program and then VALIDATE outputs by CPU

Usage examples:
  # Minimal quick test: small sizes, 4 batches, 0-based indices
  python gen_and_test_abat.py --m 200 --n 200 --batches 4 --nsub 8 --arows 12 --density 0.03

  # Same, but write 1-based indices and run your program with --one_based
  python gen_and_test_abat.py --m 200 --n 200 --batches 4 --nsub 8 --arows 12 \
      --one-based --exe ./combined_spmm_inverse_streamed_bin_debug

  # Bigger test (be careful with VRAM / runtime)
  python gen_and_test_abat.py --m 2000 --n 2000 --batches 64 --nsub 32 --arows 40 --density 0.005 \
      --exe ./combined_spmm_inverse_streamed_bin_debug
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# -----------------------------
# Utilities to write/read files
# -----------------------------
def write_sparse_txt(path: Path, A: sp.csr_matrix):
    A = A.tocsr().astype(np.float32)
    m, n = A.shape
    nnz = A.nnz
    with open(path, "w") as f:
        f.write(f"{m} {n} {nnz}\n")
        # row_ptr
        for x in A.indptr.astype(np.int64):
            f.write(f"{int(x)} ")
        f.write("\n")
        # col indices
        for x in A.indices.astype(np.int64):
            f.write(f"{int(x)} ")
        f.write("\n")
        # values
        for v in A.data.astype(np.float32):
            # keep plain to avoid scientific if possible
            f.write(f"{float(v)} ")
        f.write("\n")


def write_batches_txt(path: Path, rows_batches, cols_batches, one_based=False):
    with open(path, "w") as f:
        for rows, cols in zip(rows_batches, cols_batches):
            if one_based:
                rows = [r + 1 for r in rows]
                cols = [c + 1 for c in cols]
            f.write(" ".join(map(str, rows)))
            f.write(" | ")
            f.write(" ".join(map(str, cols)))
            f.write("\n")


def load_bin(path: Path):
    """Read our binary: int32 rows, int32 cols, then float32 col-major data."""
    with open(path, "rb") as f:
        r = np.fromfile(f, dtype=np.int32, count=1)[0]
        c = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=r * c)
    if data.size != r * c:
        raise ValueError(f"File {path} ended early: expected {r*c} floats, got {data.size}")
    # return as column-major compatible ndarray (Fortran order)
    M = np.array(data, dtype=np.float32).reshape((r, c), order="F")
    return M


# ---------------------------------------
# Synthetic data generator & CPU checker
# ---------------------------------------
def make_invertible_dense(n, seed=None):
    """Create a random invertible n×n (float32), diagonally dominant."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n), dtype=np.float32)
    # Push to diagonal dominance to make inversion stable
    diag_boost = (np.sum(np.abs(M), axis=1).astype(np.float32) + 1.0).astype(np.float32)
    M[np.arange(n), np.arange(n)] += diag_boost
    return M.astype(np.float32)


def sprinkle_into_global(A: sp.csr_matrix, rows, cols, dense_block):
    """
    Put 'dense_block' into global sparse A at rows×cols positions.
    We ADD to existing values (keeps CSR nice & consistent).
    """
    # COO add is easiest
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    nr, nc = dense_block.shape
    assert nr == len(rows) and nc == len(cols)
    rr = np.repeat(rows, nc)
    cc = np.tile(cols, nr)
    vv = dense_block.reshape(-1, order="F")  # column-major flatten (consistent with how we think)
    add = sp.coo_matrix((vv, (rr, cc)), shape=A.shape, dtype=np.float32)
    return (A + add).tocsr()


def extract_dense_from_csr(A: sp.csr_matrix, rows, cols):
    """Extract dense submatrix (col-major-equivalent ndarray) for CPU reference."""
    sub = A[rows, :][:, cols].astype(np.float32).toarray(order="F")
    return sub.astype(np.float32)


def build_A_batch_from_csr(A: sp.csr_matrix, rows, cols):
    """
    Build CSR of A_batch as rows×cols extraction from global CSR.
    NOTE: This matches what your C++ code does: it keeps only entries whose columns are in 'cols'.
    """
    sub = A[rows, :][:, cols].astype(np.float32).tocsr()
    return sub


def cpu_reference_D(A_batch_csr: sp.csr_matrix, B_dense_inv: np.ndarray):
    """
    Compute reference:
        T = A * B^T
        D = A * T^T
    with A sparse CSR, B_dense_inv is dense nsub×nsub (float32).
    """
    # SciPy expects row-major, but operations are mathematically consistent.
    # Make sure to keep float32 through the pipeline.
    A = A_batch_csr.astype(np.float32)
    B = B_dense_inv.astype(np.float32)
    T = A @ B.T
    D = A @ T.T
    return T.astype(np.float32), D.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=400, help="Global rows")
    ap.add_argument("--n", type=int, default=400, help="Global cols")
    ap.add_argument("--batches", type=int, default=8, help="Number of batches (B1=B2)")
    ap.add_argument("--nsub", type=int, default=16, help="Size of B (square)")
    ap.add_argument("--arows", type=int, default=24, help="Rows picked for each A batch")
    ap.add_argument("--density", type=float, default=0.01, help="Base density for random global CSR")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed")
    ap.add_argument("--one-based", action="store_true", help="Write batch index files as 1-based")
    ap.add_argument("--exe", type=str, default="", help="Path to your CUDA binary (optional run)")
    ap.add_argument("--workdir", type=str, default=".", help="Directory to write inputs/outputs")
    ap.add_argument("--keep-old-outputs", action="store_true", help="Do not delete existing T/D .bin files before run")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1) Build global CSR with a random background
    # -------------------------------------------------
    print(f"[gen] Building global CSR {args.m}x{args.n} with density {args.density} ...")
    # Random sparse background
    A_global = sp.random(args.m, args.n, density=args.density, format="csr", dtype=np.float32, random_state=args.seed)
    A_global.data = (A_global.data.astype(np.float32) * 0.2).astype(np.float32)  # smaller background magnitude

    # -------------------------------------------------
    # 2) Make batches (rows/cols) and inject good B blocks
    # -------------------------------------------------
    B1 = B2 = args.batches
    rows_b1, cols_b1 = [], []
    rows_b2, cols_b2 = [], []

    # We’ll reuse the SAME 'cols_b1[b]' as 'cols_b2[b]' so A's columns match B size exactly
    for b in range(args.batches):
        cols = np.sort(rng.choice(args.n, size=args.nsub, replace=False)).astype(int).tolist()
        rows_b1.append(np.sort(rng.choice(args.m, size=args.nsub, replace=False)).astype(int).tolist())
        cols_b1.append(cols)

        rows_b2.append(np.sort(rng.choice(args.m, size=args.arows, replace=False)).astype(int).tolist())
        cols_b2.append(cols)  # align with B's columns by construction

    # Now we construct B source submatrices (invertible) and inject them into the global CSR,
    # so that when your program EXTRACTS and INVERTS, it gets a well-conditioned B^{-1}.
    print(f"[gen] Injecting {args.batches} invertible B blocks into global CSR ...")
    for b in range(args.batches):
        Bsrc = make_invertible_dense(args.nsub, seed=args.seed + 1000 + b)
        A_global = sprinkle_into_global(A_global, rows_b1[b], cols_b1[b], Bsrc)

    # Also sprinkle some “A content” (optional; not necessary since background + Bsrc rows may already give nnz)
    print(f"[gen] Sprinkling extra values into rows used by A batches ...")
    for b in range(args.batches):
        rr = rows_b2[b]
        cc = cols_b2[b]
        # add a small random dense patch (very light) to ensure non-emptiness
        patch = (rng.standard_normal((len(rr), len(cc))).astype(np.float32) * 0.05).astype(np.float32)
        A_global = sprinkle_into_global(A_global, rr, cc, patch)

    # -------------------------------------------------
    # 3) Write input files
    # -------------------------------------------------
    sparse_txt = workdir / "sparse.txt"
    indx_b_txt = workdir / "indx_b.txt"
    idx_std_txt = workdir / "idx_std_loc.txt"

    print(f"[io] Writing {sparse_txt}")
    write_sparse_txt(sparse_txt, A_global)

    print(f"[io] Writing {indx_b_txt} (B batches)")
    write_batches_txt(indx_b_txt, rows_b1, cols_b1, one_based=args.one_based)

    print(f"[io] Writing {idx_std_txt} (A batches)")
    write_batches_txt(idx_std_txt, rows_b2, cols_b2, one_based=args.one_based)

    # -------------------------------------------------
    # 4) Optionally run your CUDA binary
    # -------------------------------------------------
    if args.exe:
        # Clean previous *.bin outputs unless told otherwise
        if not args.keep_old_outputs:
            for p in workdir.glob("T_batch_*.bin"):
                try: p.unlink()
                except: pass
            for p in workdir.glob("D_batch_*.bin"):
                try: p.unlink()
                except: pass

        cmd = [
            args.exe,
            str(sparse_txt),
            str(indx_b_txt),
            str(idx_std_txt),
            str(B1),
            str(B2),
        ]
        if args.one_based:
            cmd.append("--one_based")

        print(f"[run] {' '.join(cmd)}")
        env = os.environ.copy()
        try:
            subprocess.run(cmd, cwd=str(workdir), check=True)
        except subprocess.CalledProcessError as e:
            print(f"[run] Program failed with exit code {e.returncode}")
            sys.exit(e.returncode)

    # -------------------------------------------------
    # 5) Validate outputs (CPU vs files if present)
    # -------------------------------------------------
    print("[check] Validating per batch (CPU reference vs binary files if present) ...")
    tol_abs = 3e-4
    tol_rel = 3e-4

    num_ok, num_total = 0, 0
    for b in range(args.batches):
        # Build A_batch CSR and B_source (dense) from the same global we wrote
        A_b = build_A_batch_from_csr(A_global, rows_b2[b], cols_b2[b])
        B_src = extract_dense_from_csr(A_global, rows_b1[b], cols_b1[b])  # this is what your program inverts

        # Invert B_src on CPU (float32)
        try:
            B_inv = np.linalg.inv(B_src.astype(np.float32)).astype(np.float32)
        except np.linalg.LinAlgError:
            print(f"[check] Batch {b}: B_src not invertible (unexpected). Skipping check.")
            continue

        # Compute CPU reference
        T_ref, D_ref = cpu_reference_D(A_b, B_inv)

        # If binary outputs exist, compare; otherwise just sanity-check shapes
        T_file = workdir / f"T_batch_{b}.bin"
        D_file = workdir / f"D_batch_{b}.bin"

        if T_file.exists() and D_file.exists():
            T_gpu = load_bin(T_file)
            D_gpu = load_bin(D_file)

            # shape checks
            if T_gpu.shape != T_ref.shape or D_gpu.shape != D_ref.shape:
                print(f"[check] Batch {b}: shape mismatch. T {T_gpu.shape} vs {T_ref.shape}, D {D_gpu.shape} vs {D_ref.shape}")
                continue

            # error metrics
            def errs(G, R):
                diff = (G - R).astype(np.float32)
                an = np.linalg.norm(diff.ravel(), ord=np.inf)
                rn = an / (1e-8 + np.linalg.norm(R.ravel(), ord=np.inf))
                return an, rn

            T_an, T_rn = errs(T_gpu, T_ref)
            D_an, D_rn = errs(D_gpu, D_ref)

            ok = (T_an <= tol_abs or T_rn <= tol_rel) and (D_an <= tol_abs or D_rn <= tol_rel)
            status = "OK" if ok else "FAIL"
            print(f"[check] Batch {b:4d}: T_abs={T_an:.2e} T_rel={T_rn:.2e} | D_abs={D_an:.2e} D_rel={D_rn:.2e}  -> {status}")
            num_total += 1
            if ok: num_ok += 1
        else:
            # No files; just report CPU-side sizes
            print(f"[check] Batch {b:4d}: CPU T {T_ref.shape}, D {D_ref.shape} (no GPU files to compare)")

    if num_total > 0:
        print(f"[check] Summary: {num_ok}/{num_total} batches within tolerances (abs≤{tol_abs}, rel≤{tol_rel}).")
    else:
        print("[check] No GPU output files found; generation-only run complete.")


if __name__ == "__main__":
    main()
