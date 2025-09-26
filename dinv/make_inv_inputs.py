# make_inv_inputs.py
import numpy as np, argparse, os
ap=argparse.ArgumentParser()
ap.add_argument('--n',type=int,default=80)
ap.add_argument('--nbatch',type=int,default=64)
ap.add_argument('--seed',type=int,default=12345)
ap.add_argument('--outdir',type=str,default='in_inv'); args=ap.parse_args()
rng=np.random.default_rng(args.seed)
n,nb=args.n,args.nbatch
A=np.empty((nb,n,n),dtype=np.float32)
for b in range(nb):
    M=rng.standard_normal((n,n)).astype(np.float32)
    A[b]=M@M.T; A[b].flat[::n+1]+=n  # SPD + diag for conditioning
os.makedirs(args.outdir,exist_ok=True)
A.reshape(nb,-1).tofile(os.path.join(args.outdir,'A_batched_f32.bin'))
print('Wrote',os.path.join(args.outdir,'A_batched_f32.bin'))
