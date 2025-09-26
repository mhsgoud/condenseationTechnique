# verify_from_files.py — verifies D = A*B*A^T
import json, os, numpy as np, scipy.sparse as sp

def load_trimmed(path, dtype, need, label):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size < need:
        raise ValueError(f"{label}: need {need}, found {arr.size} in {path}")
    if arr.size > need:
        print(f"[WARN] {label}: {path} has {arr.size} elems; trimming to last {need}.")
        arr = arr[-need:]
    return arr

with open("out/metadata.json") as f: meta=json.load(f)
m,k,nb = int(meta["m"]), int(meta["k"]), int(meta["nbatch"])
print(f"[INFO] m={m} k={k} nbatch={nb}")

rowptr = np.fromfile("in/A_rowptr_i32.bin", dtype=np.int32)
colind = np.fromfile("in/A_colind_i32.bin", dtype=np.int32)
nnz = int(rowptr[-1])
print(f"[INFO] nnz={nnz} (density={nnz/(m*k):.3%})")

Avals = load_trimmed("in/A_vals_batched_f32.bin", np.float32, nb*nnz, "Avals").reshape(nb, nnz)
Bflat = load_trimmed("in/B_batched_f32.bin",      np.float32, nb*k*k, "B").reshape(nb, k, k)
Dflat = load_trimmed("out/D_batched_f32.bin",     np.float32, nb*m*m, "D")

# Try both interpretations for D (row-major vs col-major on disk)
D_C = Dflat.reshape(nb, m, m, order="C")
D_F = Dflat.reshape(nb, m, m, order="F")

batches = [0,1,nb-1] if nb>2 else list(range(nb))
maxC=maxF=0.0

for b in batches:
    A = sp.csr_matrix((Avals[b], colind, rowptr), shape=(m,k))
    Dcpu = (A @ Bflat[b] @ A.T).astype(np.float32)
    Dcpu = np.asarray(Dcpu)

    eC = float(np.max(np.abs(Dcpu - D_C[b])))
    eF = float(np.max(np.abs(Dcpu - D_F[b])))
    print(f"batch {b:5d}: max|Dcpu-Dgpu|  (C-map)= {eC:.3e}   (F-map)= {eF:.3e}")
    maxC = max(maxC, eC); maxF = max(maxF, eF)

print("\nSummary:")
print(f"  C/row-major map  max err = {maxC:.3e}")
print(f"  F/col-major map  max err = {maxF:.3e}")
print("✓ Mapping looks:", "C (row-major)" if maxC<=maxF else "F (column-major)")
