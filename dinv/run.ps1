nvcc -O3 -std=c++17 -arch=sm_75 invert_bench_batched.cu -lcusparse -lcublas -o invert_bench_batched.exe -I"C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen"  
python make_inv_inputs.py --n 80 --nbatch 1512 --outdir in_inv 
.\invert_bench_batched.exe --n 80 --nbatch 1512 --A in_inv\A_batched_f32.bin --repeat 5 --cpu single --outdir out_inv
python verify_inverse.py --n 80 --nbatch 1512 --dir out_inv