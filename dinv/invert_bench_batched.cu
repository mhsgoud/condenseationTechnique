// invert_bench_batched_eigen_fixed.cu
// CPU vs GPU timings for batched inverse (float32), ROW-major files on disk.
// GPU: cuBLAS getrfBatched + getriBatched (batched LU + inverse)
// CPU: Eigen (PartialPivLU -> inverse), single-thread or OpenMP (if host compiler enables it)
//
// Build (example):
//   nvcc -O3 -std=c++17 -arch=sm_75 invert_bench_batched_eigen_fixed.cu \
//        -lcublas -lcudart -I"C:/path/to/eigen-root" -o invert_bench_batched.exe
//   (optional OpenMP for Eigen) add: -Xcompiler "/openmp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

// ---- Eigen (header-only) ----
#include <C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen/Core>
#include <C:/Users/mhsgoud/Downloads/eigen-master/eigen-master/Eigen/LU>

// ---------- helpers ----------
#define CHECK_CUDA(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while(0)

static const char* cublas_err(cublasStatus_t s){
    switch(s){
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        default: return "UNKNOWN";
    }
}
#define CHECK_CUBLAS(x) do { \
    cublasStatus_t st__ = (x); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS %s:%d: %s\n", __FILE__, __LINE__, cublas_err(st__)); \
        std::exit(1); \
    } \
} while(0)

struct Args {
    int N=80, nbatch=1, repeat=5;
    std::string A_path, outdir="out_inv";
    enum CPUKind { NONE, SINGLE, OMP } cpu=SINGLE;
};
static void usage(){
    std::puts("Usage:\n  invert_bench_batched.exe --n N --nbatch B --A in/A_batched_f32.bin "
              "[--repeat R] [--cpu none|single|omp] [--outdir out_inv]");
}
static bool parse_int(const char* s, int& out){
    char* e=nullptr; long v=strtol(s,&e,10); if(!e||*e) return false; out=(int)std::max<long>(1,v); return true;
}
static Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;++i){
        std::string k=argv[i];
        auto need=[&](const char* name){ if(i+1>=argc){ std::fprintf(stderr,"Missing value for %s\n",name); usage(); std::exit(1);} return argv[++i]; };
        if      (k=="--n")       { if(!parse_int(need("--n"), a.N)) usage(), std::exit(1); }
        else if (k=="--nbatch")  { if(!parse_int(need("--nbatch"), a.nbatch)) usage(), std::exit(1); }
        else if (k=="--A")       { a.A_path = need("--A"); }
        else if (k=="--repeat")  { if(!parse_int(need("--repeat"), a.repeat)) usage(), std::exit(1); }
        else if (k=="--outdir")  { a.outdir = need("--outdir"); }
        else if (k=="--cpu")     { std::string v=need("--cpu");
                                   if(v=="none") a.cpu=Args::NONE; else if(v=="single") a.cpu=Args::SINGLE; else if(v=="omp") a.cpu=Args::OMP;
                                   else usage(), std::exit(1); }
        else { std::fprintf(stderr,"Unknown arg: %s\n", k.c_str()); usage(); std::exit(1); }
    }
    if(a.A_path.empty()){ std::fprintf(stderr,"ERROR: --A path required\n"); usage(); std::exit(1); }
    return a;
}

template<class T>
static std::vector<T> read_bin(const std::string& path, size_t expected=0){
    std::ifstream f(path, std::ios::binary);
    if(!f){ std::fprintf(stderr,"Failed to open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0,std::ios::end); size_t bytes=(size_t)f.tellg(); f.seekg(0);
    if(expected && bytes != expected*sizeof(T)){
        std::fprintf(stderr,"Size mismatch %s: got %zu, expected %zu bytes\n", path.c_str(), bytes, expected*sizeof(T));
        std::exit(1);
    }
    std::vector<T> v(bytes/sizeof(T)); if(bytes) f.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes)); return v;
}
static void write_bin(const std::string& path, const void* data, size_t bytes){
    std::ofstream f(path, std::ios::binary);
    if(!f){ std::fprintf(stderr,"Failed to open %s for write\n", path.c_str()); std::exit(1); }
    f.write(reinterpret_cast<const char*>(data), std::streamsize(bytes));
}

// ------------- CPU (Eigen) -------------
static void cpu_eigen_inverse_batched(const std::vector<float>& Arow,
                                      std::vector<float>& Ainv_row,
                                      int N, int nbatch, int repeat,
                                      Args::CPUKind kind)
{
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const size_t stride = (size_t)N*N;

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int rep=0; rep<repeat; ++rep){
        for(int b=0; b<nbatch; ++b){
            const float* Aptr = &Arow[(size_t)b*stride];
            Eigen::Map<const Mat> A(Aptr, N, N);
            Eigen::PartialPivLU<Mat> lu(A);
            Mat invA = lu.inverse();
            std::memcpy(&Ainv_row[(size_t)b*stride], invA.data(), stride*sizeof(float));
            volatile float sink = invA(0,0); (void)sink;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count() / repeat;

    std::printf("\nCPU (Eigen %s) timing over %d reps:\n",
        kind==Args::OMP ? "OpenMP" : "single-thread", repeat);
    std::printf("  Avg time (all batches): %.3f ms\n", ms);
    std::printf("  Per-batch time        : %.6f ms\n", ms / nbatch);
}

// ------------- main -------------
int main(int argc, char** argv){
    Args args = parse_args(argc, argv);
    const int N=args.N, nb=args.nbatch;
    const size_t stride=(size_t)N*N, total=(size_t)nb*stride;

    // Load A (row-major on disk)
    std::vector<float> h_A_row = read_bin<float>(args.A_path);
    if(h_A_row.size()!=total){
        std::fprintf(stderr,"ERROR: A has %zu floats, expected %zu (= nbatch*N*N)\n", h_A_row.size(), total);
        return 1;
    }
    std::printf("[INFO] Inverse benchmark: N=%d, nbatch=%d, repeat=%d (row-major files)\n",
                N, nb, args.repeat);

    // Prepare column-major copy (host) for cuBLAS (one-time transpose)
    std::vector<float> h_A_col(total);
    for(int b=0;b<nb;++b){
        const float* Ain = &h_A_row[(size_t)b*stride];
        float*       Aco = &h_A_col[(size_t)b*stride];
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
                Aco[(size_t)j*N + i] = Ain[(size_t)i*N + j];
    }

    // --- GPU setup ---
    float *d_Acol_pristine=nullptr, *d_Acol_work=nullptr, *d_Ainv_col=nullptr;
    CHECK_CUDA(cudaMalloc(&d_Acol_pristine, total*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Acol_work,     total*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ainv_col,      total*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Acol_pristine, h_A_col.data(), total*sizeof(float), cudaMemcpyHostToDevice));

    // device pointer arrays MUST point to the WORK buffer (each into its slab)
    std::vector<float*> h_Aptr(nb), h_Cptr(nb);
    for(int b=0;b<nb;++b){ h_Aptr[b]=d_Acol_work+(size_t)b*stride; h_Cptr[b]=d_Ainv_col+(size_t)b*stride; }
    float **d_Aptr=nullptr, **d_Cptr=nullptr;
    CHECK_CUDA(cudaMalloc(&d_Aptr, nb*sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_Cptr, nb*sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_Aptr, h_Aptr.data(), nb*sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Cptr, h_Cptr.data(), nb*sizeof(float*), cudaMemcpyHostToDevice));

    int *d_piv=nullptr, *d_info=nullptr;
    CHECK_CUDA(cudaMalloc(&d_piv,  (size_t)nb*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, (size_t)nb*sizeof(int)));

    cublasHandle_t H=nullptr; CHECK_CUBLAS(cublasCreate(&H));

    // --- GPU timing ---
    float incl_sum=0.f, excl_sum=0.f;
    int lu_fail_total=0, inv_fail_total=0;

    for(int rep=0; rep<args.repeat; ++rep){
        cudaEvent_t s0,e0,s1,e1; CHECK_CUDA(cudaEventCreate(&s0)); CHECK_CUDA(cudaEventCreate(&e0));
        CHECK_CUDA(cudaEventCreate(&s1)); CHECK_CUDA(cudaEventCreate(&e1));

        // INCLUDING H2D/D2H:
        // re-upload pristine from host (counts H2D), then work <- pristine (D2D), then compute, then D2H
        CHECK_CUDA(cudaEventRecord(s0));
        CHECK_CUDA(cudaMemcpy(d_Acol_pristine, h_A_col.data(), total*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Acol_work, d_Acol_pristine, total*sizeof(float), cudaMemcpyDeviceToDevice));

        CHECK_CUBLAS(cublasSgetrfBatched(H, N, d_Aptr, N, d_piv, d_info, nb));
        {
            // check LU infos (non-zero => singular/failed)
            std::vector<int> info(nb);
            CHECK_CUDA(cudaMemcpy(info.data(), d_info, nb*sizeof(int), cudaMemcpyDeviceToHost));
            for(int i=0;i<nb;++i) if(info[i]!=0) ++lu_fail_total;
        }
        CHECK_CUBLAS(cublasSgetriBatched(H, N, (const float**)d_Aptr, N, d_piv, d_Cptr, N, d_info, nb));
        {
            std::vector<int> info(nb);
            CHECK_CUDA(cudaMemcpy(info.data(), d_info, nb*sizeof(int), cudaMemcpyDeviceToHost));
            for(int i=0;i<nb;++i) if(info[i]!=0) ++inv_fail_total;
        }

        // D2H (download col-major inverse; we’ll only time it, not store it per-rep)
        std::vector<float> tmp_col(total);
        CHECK_CUDA(cudaMemcpy(tmp_col.data(), d_Ainv_col, total*sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaEventSynchronize(e0));
        float ms_incl=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms_incl, s0,e0));
        incl_sum += ms_incl;

        // EXCLUDING copies (compute only): reset work <- pristine then compute
        CHECK_CUDA(cudaEventRecord(s1));
        CHECK_CUDA(cudaMemcpy(d_Acol_work, d_Acol_pristine, total*sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasSgetrfBatched(H, N, d_Aptr, N, d_piv, d_info, nb));
        CHECK_CUBLAS(cublasSgetriBatched(H, N, (const float**)d_Aptr, N, d_piv, d_Cptr, N, d_info, nb));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms_excl=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms_excl, s1,e1));
        excl_sum += ms_excl;

        CHECK_CUDA(cudaEventDestroy(s0)); CHECK_CUDA(cudaEventDestroy(e0));
        CHECK_CUDA(cudaEventDestroy(s1)); CHECK_CUDA(cudaEventDestroy(e1));
    }

    float t_incl = incl_sum/args.repeat;
    float t_excl = excl_sum/args.repeat;
    std::printf("\nGPU timings over %d reps (nbatch=%d, N=%d):\n", args.repeat, nb, N);
    std::printf("  Avg time INCLUDING H2D/D2H : %.3f ms\n", t_incl);
    std::printf("  Avg time EXCLUDING copies  : %.3f ms\n", t_excl);
    std::printf("  Per-batch time (excl)      : %.6f ms\n", t_excl/nb);
    if(lu_fail_total || inv_fail_total){
        std::printf("  [WARN] LU info nonzero total=%d, INV info nonzero total=%d (over all reps)\n",
                    lu_fail_total, inv_fail_total);
    }

    // Pull final GPU inverse and write ROW-major result (transpose per matrix)
    std::vector<float> h_inv_col(total), h_inv_row(total);
    CHECK_CUDA(cudaMemcpy(h_inv_col.data(), d_Ainv_col, total*sizeof(float), cudaMemcpyDeviceToHost));
    for(int b=0;b<nb;++b){
        float* dst=&h_inv_row[(size_t)b*stride];
        const float* src=&h_inv_col[(size_t)b*stride];
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
                dst[(size_t)i*N + j] = src[(size_t)j*N + i];
    }
    std::filesystem::create_directories(args.outdir);
    write_bin(args.outdir + "/inv_batched_f32.bin", h_inv_row.data(), total*sizeof(float));
    write_bin(args.outdir + "/A_batched_f32.bin",   h_A_row.data(),   total*sizeof(float));
    std::printf("\n[OK] Wrote %s\\inv_batched_f32.bin (row-major)\n", args.outdir.c_str());

    // CPU baseline (Eigen)
    if(args.cpu != Args::NONE){
        std::vector<float> h_cpu_inv(total);
        cpu_eigen_inverse_batched(h_A_row, h_cpu_inv, N, nb, args.repeat, args.cpu);
    }

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(H));
    cudaFree(d_Aptr); cudaFree(d_Cptr);
    cudaFree(d_piv);  cudaFree(d_info);
    cudaFree(d_Acol_pristine); cudaFree(d_Acol_work); cudaFree(d_Ainv_col);
    return 0;
}
