#!/usr/bin/env python3
import argparse, csv, itertools, os, re, subprocess, sys, time
from pathlib import Path

# --- Regexes to parse your runtime report (and progress) on the fly ---
R = {
    'total': re.compile(r"Total:\s+([\d\.]+)\s+ms"),
    'io_read': re.compile(r"I/O read:\s+([\d\.]+)\s+ms"),
    'extract': re.compile(r"Extraction:\s+([\d\.]+)\s+ms"),
    'inv': re.compile(r"Inversion total:\s+([\d\.]+)\s+ms"),
    'getrf': re.compile(r"getrfBatched:\s+([\d\.]+)\s+ms"),
    'getri': re.compile(r"getriBatched:\s+([\d\.]+)\s+ms"),
    'sparse_prep': re.compile(r"Sparse prep:\s+([\d\.]+)\s+ms"),
    'spmm': re.compile(r"SpMM \(2 steps\):\s+([\d\.]+)\s+ms"),
    'memcpy': re.compile(r"Memcpy H2D/D2H:\s+([\d\.]+)\s+ms"),
    'write': re.compile(r"Write \.bin files:\s+([\d\.]+)\s+ms"),
    'cleanup': re.compile(r"Cleanup:\s+([\d\.]+)\s+ms"),
    'summary_nonempty': re.compile(r"Sparse A summary:\s+(\d+)\s*/\s*(\d+)"),
    'config': re.compile(r"^\[config\]\s+(.*)$"),
    'progress': re.compile(r"^\[progress\]\s+(.+)$"),
    'ws': re.compile(r"^\[ws\]\s+(.+)$"),
    'invert': re.compile(r"^\[invert\]\s+(.+)$"),
}

def parse_report_text(text):
    out = {}
    for k, rg in R.items():
        m = rg.search(text)
        if m:
            if k == 'summary_nonempty':
                out['nonempty_batches'] = int(m.group(1))
                out['batches_total'] = int(m.group(2))
            elif k in ('config','progress','ws','invert'):
                # not persisted in CSV summary; live only
                pass
            else:
                out[k] = float(m.group(1))
    return out

def parse_incremental_line(line, accum):
    """Update accum dict with any metrics found on this line; print live info."""
    line = line.rstrip()
    # echo progress/config/invert/ws lines directly
    if R['progress'].match(line) or R['config'].match(line) or R['invert'].match(line) or R['ws'].match(line):
        print(line, flush=True)
    # update summary fields if a runtime line appears
    for k, rg in R.items():
        m = rg.search(line)
        if not m: 
            continue
        if k == 'summary_nonempty':
            accum['nonempty_batches'] = int(m.group(1))
            accum['batches_total'] = int(m.group(2))
        elif k in ('config','progress','ws','invert'):
            # Ignore for CSV
            pass
        else:
            try:
                accum[k] = float(m.group(1))
            except ValueError:
                pass

def run_one(exe, gen, wrk, batches, nsub, arows, density, streams, alg, write_outputs, chunk_cap, one_based):
    wrk = Path(wrk)
    wrk.mkdir(parents=True, exist_ok=True)

    # 1) Generate inputs (no run inside generator)
    gen_cmd = [
        sys.executable, gen,
        "--m", "2000", "--n", "2000",
        "--batches", str(batches),
        "--nsub", str(nsub),
        "--arows", str(arows),
        "--density", str(density),
        "--workdir", str(wrk),
        "--keep-old-outputs"
    ]
    if one_based:
        gen_cmd.append("--one-based")
    print("[GEN]", " ".join(gen_cmd), flush=True)
    subprocess.run(gen_cmd, check=True)

    # Optionally clean GPU output files before this run
    if not write_outputs:
        for p in wrk.glob("T_batch_*.bin"): 
            try: p.unlink()
            except: pass
        for p in wrk.glob("D_batch_*.bin"): 
            try: p.unlink()
            except: pass

    # 2) Run GPU binary
    # IMPORTANT: pass file *names* and set cwd=wrk.
    args = [
        "sparse.txt",
        "indx_b.txt",
        "idx_std_loc.txt",
        str(batches),
        str(batches)
    ]
    if one_based:
        args.append("--one_based")

    env = os.environ.copy()
    env["ABAT_STREAMS"] = str(streams)
    env["ABAT_SPMM_ALG"] = alg
    env["ABAT_WRITE"] = "1" if write_outputs else "0"
    env["ABAT_CHUNK_CAP"] = str(chunk_cap)
    # live prints every batch by default (you can set to 10 for less spam)
    env.setdefault("ABAT_PROGRESS", "1")

    print("[RUN]", exe, *args, f"(cwd={wrk})", flush=True)
    t0 = time.time()
    # Use Popen to stream stdout live
    p = subprocess.Popen(
        [exe, *args],
        cwd=str(wrk),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env
    )

    accum = {}
    live_log = wrk / "last_run_live.log"
    with open(live_log, "w", encoding="utf-8") as logf:
        for line in p.stdout:
            logf.write(line)
            logf.flush()
            parse_incremental_line(line, accum)

    rc = p.wait()
    wall = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"Program failed with exit code {rc} (see {live_log})")

    # return whole stdout (from log) and metrics
    with open(live_log, "r", encoding="utf-8") as f:
        stdout_text = f.read()
    # Fill any remaining metrics from whole text
    accum.update(parse_report_text(stdout_text))
    accum["wall_s"] = wall
    return stdout_text, accum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to GPU binary")
    ap.add_argument("--gen", default="gen_and_test_abat.py", help="Path to generator script")
    ap.add_argument("--out", default="perf_results.csv")
    ap.add_argument("--wrk", default="perf_runs")
    # small test grid by default; expand later
    ap.add_argument("--quick", action="store_true", help="Run a small sanity sweep only")
    args = ap.parse_args()

    # Factor grids
    if args.quick:
        batches_list = [64]
        nsub_list    = [32, 64]
        arows_list   = [64]
        density_list = [0.005]
        streams_list = [1]
        alg_list     = ["ALG2"]
        write_list   = [False]
        chunk_caps   = [0]
        one_based    = [False]
    else:
        batches_list = [64, 256, 1024]
        nsub_list    = [32, 64, 128, 162, 256]
        arows_list   = [64, 128, 256]
        density_list = [0.002, 0.005, 0.01]
        streams_list = [1, 2]
        alg_list     = ["ALG2", "ALG1"]
        write_list   = [True, False]
        chunk_caps   = [0, 64, 128]
        one_based    = [False]

    out_csv = Path(args.out)
    # Prepare CSV (append mode; write header if file is new)
    field_order = [
        "batches","nsub","arows","density","streams","alg","write","chunk_cap","one_based",
        "nonempty_batches","batches_total","io_read","extract","inv","getrf","getri",
        "sparse_prep","spmm","memcpy","write","cleanup","total","wall_s"
    ]
    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=field_order)
            w.writeheader()

    # Sweep
    for (B, K, R, D, S, ALG, W, CAP, OB) in itertools.product(
            batches_list, nsub_list, arows_list, density_list, streams_list, alg_list, write_list, chunk_caps, one_based):
        row = {
            "batches": B, "nsub": K, "arows": R, "density": D, "streams": S, "alg": ALG,
            "write": int(W), "chunk_cap": CAP, "one_based": int(OB)
        }
        try:
            stdout, rep = run_one(args.exe, args.gen, args.wrk, B, K, R, D, S, ALG, W, CAP, OB)
            row.update(rep)
            # short one-line summary right away
            nnz_info = f"{int(rep.get('nonempty_batches',0))}/{int(rep.get('batches_total',B))}"
            total_ms = rep.get("total", float('nan'))
            spmm_ms  = rep.get("spmm", float('nan'))
            inv_ms   = rep.get("inv", float('nan'))
            print(f"[DONE] B={B} K={K} R={R} dens={D} S={S} ALG={ALG} write={int(W)} cap={CAP} | "
                  f"nonempty {nnz_info} | total {total_ms:.1f} ms, inv {inv_ms:.1f} ms, spmm {spmm_ms:.1f} ms, wall {rep.get('wall_s',0):.2f}s",
                  flush=True)
        except Exception as e:
            print(f"[ERR] B={B} K={K} R={R} dens={D} S={S} ALG={ALG} write={int(W)} cap={CAP} -> {e}", flush=True)
            # still record the params with minimal info
            row.update({"wall_s": float('nan')})

        # append this row immediately
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=field_order)
            # ensure all fields are present (missing -> None)
            for k in field_order:
                row.setdefault(k, None)
            w.writerow(row)

if __name__ == "__main__":
    main()
