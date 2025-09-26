// spmm_bench_abat.cu
// Time GPU (cuSPARSE batched SpMM) vs CPU (Eigen) for D_i = A_i * B_i * A_i^T.
// Inputs: one CSR pattern (rowptr, colind), batched A values, batched B (row-major per batch).
// Outputs: D_batched in row-major (same as your existing pipeline). Prints GFLOP/s and per-batch ms.

// Build (pick SM; set Eigen include path):
// nvcc -O3 -std=c++17 -arch=sm_75 spmm_bench_abat.cu -lcusparse -lcublas -o spmm_bench_abat.exe -I"C:/path/to/eigen"

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>

// ---- Eigen (header-only) ----
#include <C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen/Core>
#include <C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen/SparseCore>

// To let Eigen use OpenMP, compile with /openmp and do not define EIGEN_DONT_PARALLELIZE.
// For single-thread, define EIGEN_DONT_PARALLELIZE before including Eigen headers.
// #define EIGEN_DONT_PARALLELIZE

#define CHECK_CUDA(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1);} }while(0)
#define CHECK_CUSPARSE(x) do{auto s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::printf("cuSPARSE %s:%d %s\n", __FILE__, __LINE__, cusparseGetErrorString(s)); std::exit(1);} }while(0)
#define CHECK_CUBLAS(x) do{auto b=(x); if(b!=CUBLAS_STATUS_SUCCESS){ \
  std::printf("cuBLAS %s:%d %d\n", __FILE__, __LINE__, (int)b); std::exit(1);} }while(0)

struct Args {
  int m=80, k=160, nbatch=1;
  std::string rowptr, colind, avals, B, outdir="out";
  int repeat=5;              // GPU/CPU reps
  enum CPUKind { NONE, SINGLE, OMP } cpu = SINGLE;
  bool alg_csr2=false;
};
static void usage() {
  std::puts("Usage:\n  spmm_bench_abat.exe --m M --k K --nbatch N "
            "--rowptr A_rowptr_i32.bin --colind A_colind_i32.bin "
            "--avals A_vals_batched_f32.bin --B B_batched_f32.bin "
            "[--outdir out] [--repeat 5] [--cpu none|single|omp] [--alg csr2|default]");
}
static bool parse_int(const char* s, int& out){ char* e=nullptr; long v=strtol(s,&e,10); if(!e||*e) return false; out=(int)std::max<long>(1,v); return true; }
static Args parse_args(int argc, char** argv){
  Args a;
  for (int i=1;i<argc;i++){
    std::string k=argv[i];
    auto need=[&](const char* name){ if(i+1>=argc){ std::printf("Missing value for %s\n", name); usage(); std::exit(1);} return argv[++i]; };
    if(k=="--m")         { if(!parse_int(need("--m"), a.m)) {usage(); std::exit(1);} }
    else if(k=="--k")    { if(!parse_int(need("--k"), a.k)) {usage(); std::exit(1);} }
    else if(k=="--nbatch"){ if(!parse_int(need("--nbatch"), a.nbatch)) {usage(); std::exit(1);} }
    else if(k=="--rowptr"){ a.rowptr=need("--rowptr"); }
    else if(k=="--colind"){ a.colind=need("--colind"); }
    else if(k=="--avals") { a.avals =need("--avals"); }
    else if(k=="--B")     { a.B     =need("--B"); }
    else if(k=="--outdir"){ a.outdir=need("--outdir"); }
    else if(k=="--repeat"){ if(!parse_int(need("--repeat"), a.repeat)) {usage(); std::exit(1);} }
    else if(k=="--cpu")   { std::string v=need("--cpu"); 
                            if(v=="none") a.cpu=Args::NONE; else if(v=="single") a.cpu=Args::SINGLE; else if(v=="omp") a.cpu=Args::OMP;
                            else {usage(); std::exit(1);} }
    else if(k=="--alg")   { std::string v=need("--alg"); a.alg_csr2=(v=="csr2"); }
    else { std::printf("Unknown arg: %s\n", k.c_str()); usage(); std::exit(1); }
  }
  if(a.rowptr.empty()||a.colind.empty()||a.avals.empty()||a.B.empty()){
    std::puts("ERROR: must provide --rowptr --colind --avals --B"); usage(); std::exit(1);
  }
  return a;
}

template<class T>
static std::vector<T> read_bin(const std::string& path, size_t expected = 0){
  std::ifstream f(path, std::ios::binary);
  if(!f){ std::printf("Failed to open %s\n", path.c_str()); std::exit(1); }
  f.seekg(0,std::ios::end); size_t sz=(size_t)f.tellg(); f.seekg(0);
  if(expected && sz != expected*sizeof(T)){
    std::printf("Size mismatch for %s: got %zu bytes, expected %zu\n",
                path.c_str(), sz, expected*sizeof(T)); std::exit(1);
  }
  std::vector<T> v(sz/sizeof(T));
  if(sz) f.read(reinterpret_cast<char*>(v.data()), std::streamsize(sz));
  return v;
}
static void write_bin(const std::string& path, const void* data, size_t bytes){
  std::ofstream f(path, std::ios::binary);
  if(!f){ std::printf("Failed to open %s for write\n", path.c_str()); std::exit(1); }
  f.write(reinterpret_cast<const char*>(data), std::streamsize(bytes));
}

int main(int argc, char** argv){
  Args args = parse_args(argc, argv);
  const int m=args.m, k=args.k, nb=args.nbatch;

  // ---- Load inputs (row-major B per batch) ----
  std::vector<int32_t> h_rowptr = read_bin<int32_t>(args.rowptr);
  if((int)h_rowptr.size()!=m+1){ std::printf("rowptr length %zu != m+1\n", h_rowptr.size()); return 1; }
  const int nnz = h_rowptr.back();
  if(nnz<=0){ std::printf("nnz is %d (from rowptr.back()); invalid\n", nnz); return 1; }

  std::vector<int32_t> h_colind = read_bin<int32_t>(args.colind);
  if((int)h_colind.size()!=nnz){ std::printf("colind length %zu != nnz\n", h_colind.size()); return 1; }
  for(auto c: h_colind){ if(c<0 || c>=k){ std::printf("colind out of range: %d\n", (int)c); return 1; } }

  std::vector<float> h_Avals = read_bin<float>(args.avals, (size_t)nb*nnz);
  std::vector<float> h_Brm   = read_bin<float>(args.B,     (size_t)nb*(size_t)k*k);

  std::printf("[INFO] m=%d k=%d nbatch=%d nnz=%d repeat=%d cpu=%s\n",
              m,k,nb,nnz,args.repeat,
              args.cpu==Args::NONE?"none":(args.cpu==Args::SINGLE?"single":"omp"));

  // ---- GPU buffers ----
  int *d_rowptr=nullptr, *d_colind_batched=nullptr;
  float *d_Avals_all=nullptr, *d_T_all=nullptr, *d_Dt_all=nullptr, *d_D_all=nullptr, *d_B_batched=nullptr;

  CHECK_CUDA(cudaMalloc(&d_rowptr, (size_t)(m+1)*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowptr.data(), (size_t)(m+1)*sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_colind_batched, (size_t)nnz*nb*sizeof(int)));
  for(int b=0;b<nb;++b){
    CHECK_CUDA(cudaMemcpy(d_colind_batched + (size_t)b*nnz, h_colind.data(),
                          (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(cudaMalloc(&d_Avals_all, (size_t)nnz*nb*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_Avals_all, h_Avals.data(), (size_t)nnz*nb*sizeof(float), cudaMemcpyHostToDevice));

  const size_t sizeT  = (size_t)m*k;
  const size_t sizeDt = (size_t)m*m;
  CHECK_CUDA(cudaMalloc(&d_T_all,  (size_t)nb*sizeT *sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_Dt_all, (size_t)nb*sizeDt*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_D_all,  (size_t)nb*sizeDt*sizeof(float)));
  CHECK_CUDA(cudaMemset(d_T_all,  0, (size_t)nb*sizeT *sizeof(float)));
  CHECK_CUDA(cudaMemset(d_Dt_all, 0, (size_t)nb*sizeDt*sizeof(float)));
  CHECK_CUDA(cudaMemset(d_D_all,  0, (size_t)nb*sizeDt*sizeof(float)));

  CHECK_CUDA(cudaMalloc(&d_B_batched, (size_t)nb*(size_t)k*k*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_B_batched, h_Brm.data(), (size_t)nb*(size_t)k*k*sizeof(float), cudaMemcpyHostToDevice));

  // ---- Handles & descriptors ----
  cusparseHandle_t spH=nullptr; cublasHandle_t blH=nullptr;
  CHECK_CUSPARSE(cusparseCreate(&spH));
  CHECK_CUBLAS(cublasCreate(&blH));

  cusparseSpMatDescr_t A_csr;
  CHECK_CUSPARSE(cusparseCreateCsr(&A_csr, m, k, nnz,
      d_rowptr, d_colind_batched, d_Avals_all,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  // 4-arg: offsetsStride=0 (shared), batchStride=nnz (for both colind and values)
  CHECK_CUSPARSE(cusparseCsrSetStridedBatch(A_csr, nb, /*offsetsStride=*/0, /*batchStride=*/nnz));

  cusparseDnMatDescr_t B_dn, T_dn, Dt_dn;
  const int ldB = k, ldT = m, ldDt = m;
  CHECK_CUSPARSE(cusparseCreateDnMat(&B_dn,  k, k, ldB,  d_B_batched, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CHECK_CUSPARSE(cusparseCreateDnMat(&T_dn,  m, k, ldT,  d_T_all,     CUDA_R_32F, CUSPARSE_ORDER_COL));
  CHECK_CUSPARSE(cusparseCreateDnMat(&Dt_dn, m, m, ldDt, d_Dt_all,    CUDA_R_32F, CUSPARSE_ORDER_COL));
  CHECK_CUSPARSE(cusparseDnMatSetStridedBatch(B_dn,  nb, (long long)k*k));
  CHECK_CUSPARSE(cusparseDnMatSetStridedBatch(T_dn,  nb, (long long)sizeT));
  CHECK_CUSPARSE(cusparseDnMatSetStridedBatch(Dt_dn, nb, (long long)sizeDt));

  float alpha=1.f, beta=0.f;
  size_t buf1=0, buf2=0;
  auto algo = args.alg_csr2 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_ALG_DEFAULT;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_csr, B_dn, &beta, T_dn, CUDA_R_32F, algo, &buf1));
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
      &alpha, A_csr, T_dn, &beta, Dt_dn, CUDA_R_32F, algo, &buf2));
  size_t bufSz = std::max(buf1, buf2); void* dBuf=nullptr;
  if(bufSz) CHECK_CUDA(cudaMalloc(&dBuf, bufSz));

  // ---- GPU timing (repeat) ----
  float g_ms_total_incl=0.f, g_ms_total_excl=0.f;
  for(int rep=0; rep<args.repeat; ++rep){
    // incl H2D/D2H: re-upload values and B each rep
    cudaEvent_t ev0, ev1, ev2, ev3, ev4; 
    cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2); cudaEventCreate(&ev3); cudaEventCreate(&ev4);

    CHECK_CUDA(cudaEventRecord(ev0));
    CHECK_CUDA(cudaMemcpy(d_Avals_all, h_Avals.data(), (size_t)nnz*nb*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_batched, h_Brm.data(), (size_t)nb*(size_t)k*k*sizeof(float), cudaMemcpyHostToDevice));

    // Stage 1
    CHECK_CUSPARSE(cusparseSpMM(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_csr, B_dn, &beta, T_dn, CUDA_R_32F, algo, dBuf));
    // Stage 2
    CHECK_CUSPARSE(cusparseSpMM(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
      &alpha, A_csr, T_dn, &beta, Dt_dn, CUDA_R_32F, algo, dBuf));
    // Transpose Dt -> D
    for (int b=0;b<nb;++b){
      const float* src = d_Dt_all + (size_t)b*sizeDt;
      float*       dst = d_D_all  + (size_t)b*sizeDt;
      CHECK_CUBLAS(cublasSgeam(blH, CUBLAS_OP_T, CUBLAS_OP_N,
                               m, m, &alpha, src, ldDt, &beta, src, ldDt, dst, m));
    }
    // optional D2H (omit for excl-H2D/D2H timing)
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms_incl=0.f; cudaEventElapsedTime(&ms_incl, ev0, ev1);
    g_ms_total_incl += ms_incl;

    // excl H2D/D2H: only kernel region
    CHECK_CUDA(cudaEventRecord(ev2));
    // Stage 1
    CHECK_CUSPARSE(cusparseSpMM(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, A_csr, B_dn, &beta, T_dn, CUDA_R_32F, algo, dBuf));
    // Stage 2
    CHECK_CUSPARSE(cusparseSpMM(spH,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
      &alpha, A_csr, T_dn, &beta, Dt_dn, CUDA_R_32F, algo, dBuf));
    // Transpose
    for (int b=0;b<nb;++b){
      const float* src = d_Dt_all + (size_t)b*sizeDt;
      float*       dst = d_D_all  + (size_t)b*sizeDt;
      CHECK_CUBLAS(cublasSgeam(blH, CUBLAS_OP_T, CUBLAS_OP_N,
                               m, m, &alpha, src, ldDt, &beta, src, ldDt, dst, m));
    }
    CHECK_CUDA(cudaEventRecord(ev3));
    CHECK_CUDA(cudaEventSynchronize(ev3));
    float ms_excl=0.f; cudaEventElapsedTime(&ms_excl, ev2, ev3);
    g_ms_total_excl += ms_excl;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2); cudaEventDestroy(ev3); cudaEventDestroy(ev4);
  }
  float g_ms_avg_incl = g_ms_total_incl / args.repeat;
  float g_ms_avg_excl = g_ms_total_excl / args.repeat;

  // Effective FLOPs per batch:
  const double ops_per_batch = 2.0*nnz*(double)k + 2.0*nnz*(double)m; // A*B + A*T^T
  const double ops_total = ops_per_batch * nb;

  double g_gflops_incl = (ops_total / 1e9) / (g_ms_avg_incl / 1e3);
  double g_gflops_excl = (ops_total / 1e9) / (g_ms_avg_excl / 1e3);

  std::printf("\nGPU (batched) timing over %d reps:\n", args.repeat);
  std::printf("  avg time INCLUDING H2D (ms): %.3f   => %.2f GFLOP/s\n", g_ms_avg_incl, g_gflops_incl);
  std::printf("  avg time EXCLUDING H2D/D2H (ms): %.3f   => %.2f GFLOP/s\n", g_ms_avg_excl, g_gflops_excl);
  std::printf("  per-batch time excl copies (ms): %.6f\n", g_ms_avg_excl / nb);

  // ---- CPU baseline (Eigen) ----
  if (args.cpu != Args::NONE) {
    // Build sparse pattern once
    using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor, int>; // row-major sparse is OK
    std::vector<Eigen::Triplet<float,int>> trips; trips.reserve(nnz);
    for (int i=0;i<m;++i){
      int start=h_rowptr[i], end=h_rowptr[i+1];
      for (int p=start;p<end;++p) trips.emplace_back(i, (int)h_colind[p], 0.0f); // values set per batch
    }
    SpMat A(m,k); A.setFromTriplets(trips.begin(), trips.end()); // structure only

    // Map B and D (row-major) to Eigen for convenience
    using DenseRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int rep=0; rep<args.repeat; ++rep){
      // loop batches
      for(int b=0;b<nb;++b){
        // set A values for this batch (efficient: use valuePtr)
        float* AvalPtr = const_cast<float*>(A.valuePtr());
        std::memcpy(AvalPtr, h_Avals.data() + (size_t)b*nnz, nnz*sizeof(float));

        // map B[b] row-major
        DenseRM Bb = Eigen::Map<const DenseRM>(&h_Brm[(size_t)b*k*k], k, k);

        // compute D = A * Bb * A^T
        // Eigen does: (sparse * dense) -> dense; (dense * sparse^T) -> dense
        DenseRM T = (A * Bb).eval();           // m x k
        DenseRM D = (T * A.transpose()).eval();// m x m

        // optional: do something with D (we don’t write CPU D to disk here)
        (void)D;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count() / args.repeat;
    double gflops = (ops_total / 1e9) / (ms / 1e3);

    std::printf("\nCPU (Eigen, %s) timing over %d reps:\n",
      args.cpu==Args::SINGLE?"single-thread":"OpenMP", args.repeat);
    std::printf("  avg time (ms): %.3f   => %.2f GFLOP/s\n", ms, gflops);
    std::printf("  per-batch time (ms): %.6f\n", ms / nb);
  }

  // ---- Optional: write GPU D for verification (same as before) ----
  std::vector<float> h_D((size_t)nb*sizeDt);
  CHECK_CUDA(cudaMemcpy(h_D.data(), d_D_all, (size_t)nb*sizeDt*sizeof(float), cudaMemcpyDeviceToHost));
  std::filesystem::create_directories(args.outdir);
  write_bin(args.outdir + "/D_batched_f32.bin", h_D.data(), (size_t)nb*sizeDt*sizeof(float));

  // Cleanup
  if(dBuf) cudaFree(dBuf);
  cusparseDestroySpMat(A_csr);
  cusparseDestroyDnMat(B_dn);
  cusparseDestroyDnMat(T_dn);
  cusparseDestroyDnMat(Dt_dn);
  cublasDestroy(blH);
  cusparseDestroy(spH);
  cudaFree(d_rowptr);
  cudaFree(d_colind_batched);
  cudaFree(d_Avals_all);
  cudaFree(d_B_batched);
  cudaFree(d_T_all);
  cudaFree(d_Dt_all);
  cudaFree(d_D_all);

  return 0;
}
