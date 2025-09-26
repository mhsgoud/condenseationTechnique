nvcc -O3 -std=c++17 -arch=sm_75 spmm_bench_abat.cu -lcusparse -lcublas -o spmm_bench_abat.exe -I"C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen"  
python make_spmm_inputs.py --m 280 --k 260 --nbatch 1212 --density 0.2 --outdir in
.\spmm_bench_abat.exe --m 280 --k 260 --nbatch 1212 --rowptr in\A_rowptr_i32.bin --colind in\A_colind_i32.bin --avals  in\A_vals_batched_f32.bin --B in\B_batched_f32.bin   --repeat 10  --cpu single
python.exe .\verify_from_files.py
