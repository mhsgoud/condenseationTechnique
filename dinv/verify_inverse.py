# verify_inverse.py
# Verifies inv(A) by checking A @ invA ≈ I for batched row-major float32 files.

import numpy as np
import argparse, os, json, math

def load_bin(path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != math.prod(shape):
        raise ValueError(f"Size mismatch for {path}: got {arr.size}, expected {math.prod(shape)}")
    return arr.reshape(shape)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--nbatch", type=int, required=True)
ap.add_argument("--dir", type=str, default="out_inv",
                help="folder containing A_batched_f32.bin and inv_batched_f32.bin")
ap.add_argument("--sample", type=int, default=0,
                help="verify only this many random batches (0 = all)")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--rtol", type=float, default=1e-4)
ap.add_argument("--atol", type=float, default=1e-5)
ap.add_argument("--print_batches", type=int, nargs="*", default=[0,1,-1],
                help="batches to print detailed errors for; -1 means last batch")
args = ap.parse_args()

n = args.n
nb = args.nbatch
A  = load_bin(os.path.join(args.dir, "A_batched_f32.bin"),   np.float32, (nb, n, n))
Ai = load_bin(os.path.join(args.dir, "inv_batched_f32.bin"), np.float32, (nb, n, n))

# choose which batches to check
idx = np.arange(nb)
if args.sample and args.sample < nb:
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(nb, size=args.sample, replace=False)
idx = np.unique(idx)

# stats
max_abs_err_overall = 0.0
max_rel_err_overall = 0.0

def rel_err(X, Y, eps=1e-20):
    num = np.linalg.norm(X - Y, ord='fro')
    den = np.linalg.norm(Y, ord='fro') + eps
    return float(num / den)

def cond_est(Ai_batch):
    # crude 1-norm condition estimate using invA and ||A||_1*||invA||_1
    return None  # keep simple; can add if needed

# verify
for b in idx:
    Ab  = A[b]
    Aib = Ai[b]
    Iest = Ab @ Aib
    abs_err = float(np.max(np.abs(Iest - np.eye(n, dtype=np.float32))))
    rerr    = rel_err(Iest, np.eye(n, dtype=np.float32))
    max_abs_err_overall = max(max_abs_err_overall, abs_err)
    max_rel_err_overall = max(max_rel_err_overall, rerr)

# report some specific batches (0,1,last by default)
to_print = []
for v in args.print_batches:
    if v == -1:
        to_print.append(nb-1)
    elif 0 <= v < nb:
        to_print.append(v)
to_print = np.unique(to_print)

for b in to_print:
    Ab  = A[b]
    Aib = Ai[b]
    Iest = Ab @ Aib
    abs_err = float(np.max(np.abs(Iest - np.eye(n, dtype=np.float32))))
    rerr    = rel_err(Iest, np.eye(n, dtype=np.float32))
    print(f"batch {b:5d}: max|A*invA - I| = {abs_err:.3e}   relF = {rerr:.3e}")

print(f"\n[DONE] overall max abs error = {max_abs_err_overall:.3e}   overall max relF = {max_rel_err_overall:.3e}")

# pass/fail suggestion
ok = (max_abs_err_overall <= args.atol + args.rtol)
print("PASS" if ok else "WARN: errors exceed default tolerances (consider conditioning).")
