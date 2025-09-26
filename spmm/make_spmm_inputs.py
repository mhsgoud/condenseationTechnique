# make_spmm_inputs.py — robust input generator for D = A * B * A^T
import numpy as np, os, json, argparse, scipy.sparse as sp

def write_bin(path, arr):
    # Always overwrite (truncate)
    with open(path, "wb") as f:
        arr.tofile(f)

def csr_sorted_unique(csr: sp.csr_matrix):
    """Ensure CSR has sorted, unique column indices per row (cuSPARSE-friendly)."""
    csr.sum_duplicates()
    csr.sort_indices()
    return csr

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=80)
ap.add_argument("--k", type=int, default=160)
ap.add_argument("--nbatch", type=int, default=512)
ap.add_argument("--density", type=float, default=0.20, help="nnz/(m*k)")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--outdir", type=str, default="in")
ap.add_argument("--ensure-row-nnz", action="store_true",
                help="Guarantee every A row has at least one nonzero (adds tiny diagonal if needed).")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
m, k, nb = args.m, args.k, args.nbatch

# 1) Build one CSR sparsity pattern (shared across batches)
Apat = sp.random(m, k, density=args.density, format="csr", random_state=rng, dtype=np.float32)
Apat = csr_sorted_unique(Apat)

if args.ensure_row_nnz:
    # Add tiny diagonal to empty rows only (without changing density much)
    empty = np.diff(Apat.indptr) == 0
    if empty.any():
        diag_cols = np.minimum(np.arange(m, dtype=np.int32), k - 1)
        mask = empty & (diag_cols < k)
        bump = sp.csr_matrix((np.ones(mask.sum(), np.float32),
                              diag_cols[mask], np.concatenate([[0], np.cumsum(mask)])),
                             shape=(m, k), dtype=np.float32)
        Apat = csr_sorted_unique((Apat + bump).astype(np.float32))

rowptr = Apat.indptr.astype(np.int32, copy=False)
colind = Apat.indices.astype(np.int32, copy=False)
nnz = int(colind.size)

# 2) Batched A values (one value vector per batch, same pattern)
Avals = rng.standard_normal((nb, nnz), dtype=np.float32)

# 3) Batched dense B (k×k), row-major on disk
B = rng.standard_normal((nb, k, k), dtype=np.float32)

# 4) Write everything (overwrite)
os.makedirs(args.outdir, exist_ok=True)
write_bin(os.path.join(args.outdir, "A_rowptr_i32.bin"), rowptr)
write_bin(os.path.join(args.outdir, "A_colind_i32.bin"), colind)
write_bin(os.path.join(args.outdir, "A_vals_batched_f32.bin"), Avals.reshape(-1))
write_bin(os.path.join(args.outdir, "B_batched_f32.bin"),      B.reshape(nb, -1))

meta = {
    "m": m, "k": k, "nbatch": nb,
    "nnz": nnz, "density": nnz / (m * k),
    "seed": args.seed, "order": "row",     # files are row-major
    "ensure_row_nnz": bool(args.ensure_row_nnz),
}
with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"[OK] wrote inputs in {args.outdir}/")
print(f"     m={m} k={k} nbatch={nb} nnz={nnz} (density={meta['density']:.3%}) order=row")

# make_spmm_inputs.py  (only the bottom part changed)
import json, os
# ... after you compute meta = {...} ...

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# NEW: also write the same metadata to out/
os.makedirs("out", exist_ok=True)
with open(os.path.join("out", "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"[OK] wrote inputs in {args.outdir}/")
print(f"[OK] wrote mirror metadata in out/metadata.json")
