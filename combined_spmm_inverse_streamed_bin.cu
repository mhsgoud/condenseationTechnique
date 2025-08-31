// combined_spmm_inverse_streamed_bin_env.cu
// Fast D = A * B^T * A^T per batch using cuSPARSE SpMM twice.
// - Chunked inversion of dense B batches (cuBLAS getrf/getri batched)
// - 1..N CUDA streams, grow-only reused buffers
// - Optional binary outputs (T and D)
// - Diagnostics + live progress with ETA
//
// Build:
//   nvcc -O3 -std=c++17 combined_spmm_inverse_streamed_bin_env.cu \
//        -lcublas -lcusparse -lcudart -o combined_spmm_inverse_streamed_bin_env
//
// Usage:
//   ./combined_spmm_inverse_streamed_bin_env sparse.txt indx_b.txt idx_std_loc.txt B1 B2 [--one_based]
//
// Env controls (all optional):
//   ABAT_STREAMS   : int   (default 2)           # CUDA streams to pipeline batches
//   ABAT_SPMM_ALG  : str   (ALG1|ALG2|DEFAULT)   # cuSPARSE SpMM algorithm
//   ABAT_WRITE     : int   (1=write, 0=skip)     # write T/D .bin files
//   ABAT_CHUNK_CAP : int   (0=auto, else cap)    # cap inversion chunk size
//   ABAT_PROGRESS  : int   (0=off, N=every N)    # progress print frequency (batches)

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); std::exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)st); std::exit(1);} } while(0)
#define CHECK_CUSPARSE(x) do { cusparseStatus_t st=(x); if(st!=CUSPARSE_STATUS_SUCCESS){ \
  fprintf(stderr,"cuSPARSE error %s:%d: %d\n",__FILE__,__LINE__,(int)st); std::exit(1);} } while(0)

struct RuntimeReport {
  double total_ms=0, io_read_ms=0, extract_ms=0,
         inv_ms=0, getrf_ms=0, getri_ms=0,
         sparse_prep_ms=0, spmm_ms=0,
         memcpy_ms=0, write_ms=0, cleanup_ms=0;
  void print() const {
    std::cout << "\n=== RUNTIME REPORT ===\n";
    std::cout << "Total:             " << total_ms << " ms\n";
    std::cout << "I/O read:          " << io_read_ms << " ms\n";
    std::cout << "Extraction:        " << extract_ms << " ms\n";
    std::cout << "Inversion total:   " << inv_ms << " ms\n";
    std::cout << "  getrfBatched:    " << getrf_ms << " ms\n";
    std::cout << "  getriBatched:    " << getri_ms << " ms\n";
    std::cout << "Sparse prep:       " << sparse_prep_ms << " ms\n";
    std::cout << "SpMM (2 steps):    " << spmm_ms << " ms\n";
    std::cout << "Memcpy H2D/D2H:    " << memcpy_ms << " ms\n";
    std::cout << "Write .bin files:  " << write_ms << " ms\n";
    std::cout << "Cleanup:           " << cleanup_ms << " ms\n";
    std::cout << "======================\n";
  }
};

static inline double now_ms() {
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double,std::milli>(clk::now().time_since_epoch()).count();
}

// ---------- I/O helpers ----------
static void read_sparse_matrix(const std::string &fname,int &m,int &n,int &nnz,
                               std::vector<int> &row_ptr,std::vector<int> &cols,std::vector<float> &vals,
                               double& io_ms)
{
  auto st=std::chrono::high_resolution_clock::now();
  std::ifstream fin(fname);
  if(!fin){ std::cerr<<"Cannot open "<<fname<<"\n"; std::exit(1); }
  fin>>m>>n>>nnz;
  row_ptr.resize(m+1);
  for(int i=0;i<=m;i++) fin>>row_ptr[i];
  cols.resize(nnz);
  for(int i=0;i<nnz;i++) fin>>cols[i];
  vals.resize(nnz);
  for(int i=0;i<nnz;i++) fin>>vals[i];
  io_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-st).count();
}

static void read_batches_file(const std::string &fname,int B,
                              std::vector<std::vector<int>> &row_batches,
                              std::vector<std::vector<int>> &col_batches,
                              double& io_ms)
{
  auto st=std::chrono::high_resolution_clock::now();
  row_batches.clear(); col_batches.clear();
  std::ifstream fin(fname);
  if(!fin){ std::cerr<<"Cannot open "<<fname<<"\n"; std::exit(1); }
  std::string line; int b=0;
  while(std::getline(fin,line)){
    if(line.empty()) continue;
    size_t sep=line.find('|');
    if(sep==std::string::npos){ std::cerr<<"Missing | in "<<fname<<"\n"; std::exit(1); }
    std::string left=line.substr(0,sep), right=line.substr(sep+1);
    std::stringstream lss(left), rss(right);
    std::vector<int> rows, cols;
    int v;
    while(lss>>v) rows.push_back(v);
    while(rss>>v) cols.push_back(v);
    row_batches.push_back(std::move(rows));
    col_batches.push_back(std::move(cols));
    b++;
  }
  if(b!=B) std::cerr<<"Warning: expected "<<B<<" batches in "<<fname<<", found "<<b<<"\n";
  io_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-st).count();
}

// Extract dense submatrix (col-major)
static std::vector<float> extract_dense_colmajor(
  const std::vector<int> &row_ptr,const std::vector<int> &cols,const std::vector<float> &vals,
  const std::vector<int> &row_idx,const std::vector<int> &col_idx,double& extract_ms)
{
  auto st=std::chrono::high_resolution_clock::now();
  int nr=(int)row_idx.size(), nc=(int)col_idx.size();
  if(nr!=nc){ std::cerr<<"Submatrix not square ("<<nr<<"x"<<nc<<")\n"; std::exit(1); }
  int maxc = nc? *std::max_element(col_idx.begin(),col_idx.end()) : -1;
  std::vector<int> cmap(maxc+1,-1);
  for(int j=0;j<nc;j++) cmap[col_idx[j]] = j;
  std::vector<float> dense((size_t)nr*nc,0.0f);
  for(int r=0;r<nr;r++){
    int gr=row_idx[r];
    for(int p=row_ptr[gr]; p<row_ptr[gr+1]; ++p){
      int gc=cols[p]; if(gc>=0 && gc<=maxc){ int cc=cmap[gc]; if(cc!=-1) dense[(size_t)cc*nr+r]=vals[p]; }
    }
  }
  extract_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-st).count();
  return dense;
}

// Binary writer: header (int32 rows, int32 cols) + float32 column-major data
static void write_matrix_bin(const std::string& fname,
                             const float* M, int rows, int cols, int ld,
                             double& write_ms)
{
  auto t0=std::chrono::high_resolution_clock::now();
  FILE* f = fopen(fname.c_str(),"wb");
  if(!f){ std::cerr<<"Failed to open "<<fname<<" for writing\n"; std::exit(1); }
  int32_t r=rows, c=cols;
  fwrite(&r, sizeof(int32_t), 1, f);
  fwrite(&c, sizeof(int32_t), 1, f);
  for(int j=0;j<cols;++j){
    const float* colp = M + (size_t)j*ld;
    fwrite(colp, sizeof(float), rows, f);
  }
  fclose(f);
  write_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
}

// Memory-driven chunk size for inversion
static int compute_chunk_inversion(size_t freeBytes, int nsub){
  size_t perMat = (size_t)nsub*nsub*sizeof(float)*2; // d_Ablock + d_Cblock (rough)
  size_t usable = (size_t)(freeBytes*0.6);
  if(usable < perMat) return 1;
  size_t cap = usable / perMat;
  if(cap==0) cap=1;
  return (int)std::min<size_t>(cap, 1024);
}

int main(int argc, char** argv){
  RuntimeReport report;
  auto T0 = std::chrono::high_resolution_clock::now();

  if(argc<6){
    std::cerr<<"Usage: "<<argv[0]<<" sparse.txt indx_b.txt idx_std_loc.txt B1 B2 [--one_based]\n";
    return 1;
  }
  std::string sparse_file = argv[1];
  std::string indx_b_file  = argv[2];
  std::string idx_std_file = argv[3];
  int B1 = std::stoi(argv[4]);
  int B2 = std::stoi(argv[5]);

  bool one_based = false;
  for (int i = 6; i < argc; ++i) {
    std::string opt = argv[i];
    if (opt == "--one_based") one_based = true;
  }

  // ---- Env controls ----
  auto getenv_int = [](const char* k, int def){
      if (const char* v = std::getenv(k)) return std::atoi(v);
      return def;
  };
  auto getenv_str = [](const char* k, const char* def){
      if (const char* v = std::getenv(k)) return v;
      return def;
  };
  int NSTREAMS = std::max(1, getenv_int("ABAT_STREAMS", 2));
  std::string ALG = getenv_str("ABAT_SPMM_ALG", "ALG2"); // ALG1|ALG2|DEFAULT
  bool WRITE_BIN = getenv_int("ABAT_WRITE", 1) != 0;
  int CHUNK_CAP = getenv_int("ABAT_CHUNK_CAP", 0);
  int PROGRESS_EVERY = std::max(0, getenv_int("ABAT_PROGRESS", 10));

  // Force unbuffered stdout for live progress
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::cout.setf(std::ios::unitbuf);

  std::cout << "[config] NSTREAMS=" << NSTREAMS
            << " ALG=" << ALG
            << " WRITE_BIN=" << (WRITE_BIN?1:0)
            << " CHUNK_CAP=" << CHUNK_CAP
            << " PROGRESS_EVERY=" << PROGRESS_EVERY
            << " one_based=" << (one_based?1:0)
            << "\n";

  // Read global CSR
  int m_global,n_global,nnz_global;
  std::vector<int> h_row_ptr, h_cols;
  std::vector<float> h_vals;
  read_sparse_matrix(sparse_file, m_global,n_global,nnz_global, h_row_ptr,h_cols,h_vals, report.io_read_ms);
  std::cout<<"Read sparse matrix m="<<m_global<<", n="<<n_global<<", nnz="<<nnz_global<<"\n";

  // Read batches
  std::vector<std::vector<int>> rows_b1, cols_b1, rows_b2, cols_b2;
  read_batches_file(indx_b_file, B1, rows_b1, cols_b1, report.io_read_ms);
  read_batches_file(idx_std_file, B2, rows_b2, cols_b2, report.io_read_ms);

  auto shift_to_zero_based = [&](std::vector<std::vector<int>>& rr,
                                 std::vector<std::vector<int>>& cc){
    if (!one_based) return;
    for (auto& v : rr) for (int& x : v) --x;
    for (auto& v : cc) for (int& x : v) --x;
  };
  shift_to_zero_based(rows_b1, cols_b1);
  shift_to_zero_based(rows_b2, cols_b2);

  // Extract dense B submatrices (to invert)
  std::vector<std::vector<float>> dense_list; dense_list.reserve(B1);
  int nsub=-1;
  for(int b=0;b<B1;b++){
    if(rows_b1[b].size()!=cols_b1[b].size()){
      std::cerr<<"indx_b batch "<<b<<" not square\n"; return 1;
    }
    if(nsub==-1) nsub=(int)rows_b1[b].size();
    else if((int)rows_b1[b].size()!=nsub){ std::cerr<<"indx_b batch "<<b<<" size mismatch\n"; return 1; }
    dense_list.push_back(extract_dense_colmajor(h_row_ptr,h_cols,h_vals, rows_b1[b],cols_b1[b], report.extract_ms));
  }
  if(nsub<=0){ std::cerr<<"No submatrices\n"; return 1; }
  std::cout<<"Extracted "<<B1<<" dense submatrices of size "<<nsub<<"x"<<nsub<<"\n";

  // Invert B in chunks -> host pool hInv
  const size_t matElems = (size_t)nsub*nsub;
  std::vector<float> hInv(B1 * matElems);

  cublasHandle_t cublasH; CHECK_CUBLAS(cublasCreate(&cublasH));
  size_t freeB,totalB; CHECK_CUDA(cudaMemGetInfo(&freeB,&totalB));
  int chunk = compute_chunk_inversion(freeB, nsub);
  if (CHUNK_CAP > 0) chunk = std::min(chunk, CHUNK_CAP);
  std::cout << "[invert] chunk=" << chunk
            << " (free=" << (freeB/1024/1024) << " MiB, cap=" << CHUNK_CAP << ")\n";

  cudaEvent_t ev0,ev1; CHECK_CUDA(cudaEventCreate(&ev0)); CHECK_CUDA(cudaEventCreate(&ev1));
  for(int base=0;base<B1;base+=chunk){
    int cur = std::min(chunk, B1-base);
    float* hA = nullptr; CHECK_CUDA(cudaMallocHost((void**)&hA, cur*matElems*sizeof(float)));
    for(int i=0;i<cur;i++){
      memcpy(hA + (size_t)i*matElems, dense_list[base+i].data(), matElems*sizeof(float));
    }
    float *dA=nullptr,*dC=nullptr; float **dAarr=nullptr, **dCarr=nullptr;
    int *dPiv=nullptr,*dInfo=nullptr;
    CHECK_CUDA(cudaMalloc(&dA,  cur*matElems*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC,  cur*matElems*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA,hA, cur*matElems*sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float*> hAptr(cur), hCptr(cur);
    for(int i=0;i<cur;i++){ hAptr[i]=dA+(size_t)i*matElems; hCptr[i]=dC+(size_t)i*matElems; }
    CHECK_CUDA(cudaMalloc(&dAarr, cur*sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&dCarr, cur*sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(dAarr,hAptr.data(),cur*sizeof(float*),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dCarr,hCptr.data(),cur*sizeof(float*),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&dPiv, cur*nsub*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dInfo,cur*sizeof(int)));

    float ms=0.f;
    CHECK_CUDA(cudaEventRecord(ev0));
    CHECK_CUBLAS(cublasSgetrfBatched(cublasH, nsub, dAarr, nsub, dPiv, dInfo, cur));
    CHECK_CUDA(cudaEventRecord(ev1)); CHECK_CUDA(cudaEventSynchronize(ev1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1)); report.getrf_ms += ms;

    CHECK_CUDA(cudaEventRecord(ev0));
    CHECK_CUBLAS(cublasSgetriBatched(cublasH, nsub,(const float**)dAarr,nsub,dPiv,dCarr,nsub,dInfo,cur));
    CHECK_CUDA(cudaEventRecord(ev1)); CHECK_CUDA(cudaEventSynchronize(ev1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1)); report.getri_ms += ms;

    report.inv_ms = report.getrf_ms + report.getri_ms;

    auto cp0=std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(hInv.data() + (size_t)base*matElems, dC, cur*matElems*sizeof(float), cudaMemcpyDeviceToHost));
    report.memcpy_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-cp0).count();

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dAarr)); CHECK_CUDA(cudaFree(dCarr));
    CHECK_CUDA(cudaFree(dPiv)); CHECK_CUDA(cudaFree(dInfo));
    CHECK_CUDA(cudaFreeHost(hA));
    std::cout << "[invert] done chunk base="<<base<<" count="<<cur<<"\n";
  }
  CHECK_CUDA(cudaEventDestroy(ev0)); CHECK_CUDA(cudaEventDestroy(ev1));
  cublasDestroy(cublasH);

  // Build CSR A batches with diagnostics
  auto sp_st = std::chrono::high_resolution_clock::now();
  std::vector<int> A_rows(B2), A_cols(B2), A_nnz(B2);
  std::vector<std::vector<int>> A_ofs(B2), A_col(B2);
  std::vector<std::vector<float>> A_val(B2);

  int nonempty_batches = 0, printed = 0;

  for (int b = 0; b < B2; ++b) {
    const auto& rb = rows_b2[b];
    const auto& cb = cols_b2[b];

    if (rb.empty() || cb.empty()) {
      A_rows[b] = (int)rb.size();
      A_cols[b] = (int)cb.size();
      A_nnz[b]  = 0;
      if (printed < 20) {
        std::cout << "[A-batch " << b << "] rows=" << A_rows[b]
                  << " cols=" << A_cols[b] << " nnz=0 (empty rows/cols)\n";
        printed++;
      }
      continue;
    }

    int rows = (int)rb.size();
    int cols = (int)cb.size();
    A_rows[b] = rows; A_cols[b] = cols;

    int maxc = *std::max_element(cb.begin(), cb.end());
    std::vector<int> cmap(std::max(0,maxc)+1, -1);
    for (int j = 0; j < cols; ++j) {
      int cj = cb[j];
      if (cj >= 0 && cj <= maxc) cmap[cj] = j;
    }

    A_ofs[b].resize(rows+1);
    A_col[b].clear(); A_val[b].clear();
    int acc = 0;
    for (int r = 0; r < rows; ++r) {
      int gr = rb[r];
      A_ofs[b][r] = acc;
      if (gr >= 0 && gr < m_global) {
        for (int p = h_row_ptr[gr]; p < h_row_ptr[gr+1]; ++p) {
          int gc = h_cols[p];
          if (gc >= 0 && gc <= maxc) {
            int cc = cmap[gc];
            if (cc != -1) { A_col[b].push_back(cc); A_val[b].push_back(h_vals[p]); acc++; }
          }
        }
      }
    }
    A_ofs[b][rows] = acc;
    A_nnz[b] = acc;
    if (rows > 0 && acc > 0) nonempty_batches++;

    if (printed < 20) {
      std::cout << "[A-batch " << b << "] rows="<<rows<<" cols="<<cols<<" nnz="<<acc<<"\n";
      printed++;
    }
  }
  report.sparse_prep_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-sp_st).count();
  std::cout << "Sparse A summary: " << nonempty_batches << " / " << B2
            << " batches have rows>0 and nnz>0.\n";

  // Robust probe
  int probe = -1;
  for (int i = 0; i < B2; ++i) {
    if (A_rows[i] > 0 && A_nnz[i] > 0) { probe = i; break; }
  }
  if (probe < 0) {
    std::cerr << "All idx_std_loc batches are empty (rows==0 or nnz==0).\n";
    report.total_ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-T0).count();
    report.print();
    return 0;
  }

  // cuSPARSE handle
  cusparseHandle_t cusH; CHECK_CUSPARSE(cusparseCreate(&cusH));

  // Choose SpMM algorithm from env
  cusparseSpMMAlg_t spmmAlg = CUSPARSE_SPMM_CSR_ALG2;
  if (ALG == "ALG1")     spmmAlg = CUSPARSE_SPMM_CSR_ALG1;
  if (ALG == "DEFAULT")  spmmAlg = CUSPARSE_SPMM_ALG_DEFAULT;

  float alpha=1.0f, beta=0.0f;

  // Workspace probe with the probe batch
  size_t ws_max = 0;
  {
    int rows=A_rows[probe], cols=A_cols[probe], nnz=A_nnz[probe];
    int *dro=nullptr,*dci=nullptr; float *dva=nullptr,*dB=nullptr,*dT=nullptr,*dD=nullptr;
    CHECK_CUDA(cudaMalloc(&dro,sizeof(int)*(rows+1)));
    CHECK_CUDA(cudaMalloc(&dci,sizeof(int)*nnz));
    CHECK_CUDA(cudaMalloc(&dva,sizeof(float)*nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*nsub*nsub));
    CHECK_CUDA(cudaMalloc(&dT, sizeof(float)*(size_t)rows*nsub));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(float)*(size_t)rows*rows));
    cusparseSpMatDescr_t A; cusparseDnMatDescr_t Bm,Tm,Dm;
    CHECK_CUSPARSE(cusparseCreateCsr(&A, rows, cols, nnz, dro,dci,dva, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Bm, nsub,nsub,nsub, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Tm, rows,nsub,rows, dT, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Dm, rows,rows,rows, dD, CUDA_R_32F, CUSPARSE_ORDER_COL));
    size_t ws1=0, ws2=0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &alpha, A,Bm,&beta,Tm, CUDA_R_32F, spmmAlg, &ws1));
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &alpha, A,Tm,&beta,Dm, CUDA_R_32F, spmmAlg, &ws2));
    ws_max = std::max(ws1,ws2);
    cusparseDestroySpMat(A); cusparseDestroyDnMat(Bm); cusparseDestroyDnMat(Tm); cusparseDestroyDnMat(Dm);
    CHECK_CUDA(cudaFree(dro)); CHECK_CUDA(cudaFree(dci)); CHECK_CUDA(cudaFree(dva));
    CHECK_CUDA(cudaFree(dB));  CHECK_CUDA(cudaFree(dT));  CHECK_CUDA(cudaFree(dD));
  }
  std::cout << "[ws] SpMM workspace per stream: " << (ws_max/1024.0/1024.0) << " MiB (alg="<<ALG<<")\n";

  // Streams + buffers
  std::vector<cudaStream_t> streams(NSTREAMS);
  for(int s=0;s<NSTREAMS;s++){ CHECK_CUDA(cudaStreamCreate(&streams[s])); }

  std::vector<int*> d_ro(NSTREAMS,nullptr), d_ci(NSTREAMS,nullptr);
  std::vector<float*> d_va(NSTREAMS,nullptr), d_B(NSTREAMS,nullptr), d_T(NSTREAMS,nullptr), d_D(NSTREAMS,nullptr);
  std::vector<void*> dWs(NSTREAMS,nullptr);
  for(int s=0;s<NSTREAMS;s++){ CHECK_CUDA(cudaMalloc(&dWs[s], ws_max)); }

  std::vector<float*> hT_pin(NSTREAMS,nullptr), hD_pin(NSTREAMS,nullptr);
  std::vector<size_t> hT_bytes(NSTREAMS,0), hD_bytes(NSTREAMS,0);
  std::vector<int> cap_rows(NSTREAMS,0), cap_nnz(NSTREAMS,0);
  std::vector<size_t> cap_B(NSTREAMS,0), cap_T(NSTREAMS,0), cap_D(NSTREAMS,0);

  std::vector<cudaEvent_t> ev_s1s(NSTREAMS), ev_s1e(NSTREAMS), ev_s2s(NSTREAMS), ev_s2e(NSTREAMS);
  for(int s=0;s<NSTREAMS;s++){
    CHECK_CUDA(cudaEventCreate(&ev_s1s[s])); CHECK_CUDA(cudaEventCreate(&ev_s1e[s]));
    CHECK_CUDA(cudaEventCreate(&ev_s2s[s])); CHECK_CUDA(cudaEventCreate(&ev_s2e[s]));
  }

  // Process all batches; skip empties
  int processed_batches = 0;
  double t_start_ms = now_ms();

  for(int b=0; b<B2; ++b){
    int rows=A_rows[b], cols=A_cols[b], nnz=A_nnz[b];
    if (rows==0 || nnz==0) continue; // skip empty

    int s = b % NSTREAMS;
    CHECK_CUDA(cudaStreamSynchronize(streams[s])); // ensure previous batch on this stream finished

    // (Re)alloc grow-only device buffers
    if(rows > cap_rows[s]){ if(d_ro[s]) CHECK_CUDA(cudaFree(d_ro[s])); CHECK_CUDA(cudaMalloc(&d_ro[s], sizeof(int)*(rows+1))); cap_rows[s]=rows; }
    if(nnz  > cap_nnz[s]){ if(d_ci[s]) CHECK_CUDA(cudaFree(d_ci[s])); if(d_va[s]) CHECK_CUDA(cudaFree(d_va[s]));
                            CHECK_CUDA(cudaMalloc(&d_ci[s], sizeof(int)*nnz));
                            CHECK_CUDA(cudaMalloc(&d_va[s], sizeof(float)*nnz));
                            cap_nnz[s]=nnz; }
    size_t needB = (size_t)nsub*nsub*sizeof(float);
    size_t needT = (size_t)rows*nsub*sizeof(float);
    size_t needD = (size_t)rows*rows*sizeof(float);
    if(needB > cap_B[s]){ if(d_B[s]) CHECK_CUDA(cudaFree(d_B[s])); CHECK_CUDA(cudaMalloc(&d_B[s], needB)); cap_B[s]=needB; }
    if(needT > cap_T[s]){ if(d_T[s]) CHECK_CUDA(cudaFree(d_T[s])); CHECK_CUDA(cudaMalloc(&d_T[s], needT)); cap_T[s]=needT; }
    if(needD > cap_D[s]){ if(d_D[s]) CHECK_CUDA(cudaFree(d_D[s])); CHECK_CUDA(cudaMalloc(&d_D[s], needD)); cap_D[s]=needD; }

    if(needT > hT_bytes[s]){ if(hT_pin[s]) CHECK_CUDA(cudaFreeHost(hT_pin[s])); CHECK_CUDA(cudaMallocHost((void**)&hT_pin[s], needT)); hT_bytes[s]=needT; }
    if(needD > hD_bytes[s]){ if(hD_pin[s]) CHECK_CUDA(cudaFreeHost(hD_pin[s])); CHECK_CUDA(cudaMallocHost((void**)&hD_pin[s], needD)); hD_bytes[s]=needD; }

    // H2D: A CSR + B (inverse from host pool position b)
    auto cp0=std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpyAsync(d_ro[s], A_ofs[b].data(), sizeof(int)*(rows+1), cudaMemcpyHostToDevice, streams[s]));
    if(nnz){
      CHECK_CUDA(cudaMemcpyAsync(d_ci[s], A_col[b].data(), sizeof(int)*nnz,   cudaMemcpyHostToDevice, streams[s]));
      CHECK_CUDA(cudaMemcpyAsync(d_va[s], A_val[b].data(), sizeof(float)*nnz, cudaMemcpyHostToDevice, streams[s]));
    }
    CHECK_CUDA(cudaMemcpyAsync(d_B[s],  hInv.data() + (size_t)b*matElems, needB, cudaMemcpyHostToDevice, streams[s]));
    CHECK_CUDA(cudaMemsetAsync(d_T[s], 0, needT, streams[s]));
    CHECK_CUDA(cudaMemsetAsync(d_D[s], 0, needD, streams[s]));
    report.memcpy_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-cp0).count();

    // Descriptors
    cusparseSpMatDescr_t A; cusparseDnMatDescr_t Bm, Tm, Dm;
    CHECK_CUSPARSE(cusparseCreateCsr(&A, rows, cols, nnz, d_ro[s],d_ci[s],d_va[s],
                                     CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Bm, nsub,nsub,nsub, d_B[s], CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Tm, rows,nsub,rows, d_T[s], CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Dm, rows,rows,rows, d_D[s], CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseSetStream(cusH, streams[s]));

    // SpMM #1: T = A * B^T
    CHECK_CUDA(cudaEventRecord(ev_s1s[s], streams[s]));
    CHECK_CUSPARSE(cusparseSpMM(cusH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, A,Bm,&beta,Tm, CUDA_R_32F, spmmAlg, dWs[s]));
    CHECK_CUDA(cudaEventRecord(ev_s1e[s], streams[s]));

    // SpMM #2: D = A * T^T
    CHECK_CUDA(cudaEventRecord(ev_s2s[s], streams[s]));
    CHECK_CUSPARSE(cusparseSpMM(cusH, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, A,Tm,&beta,Dm, CUDA_R_32F, spmmAlg, dWs[s]));
    CHECK_CUDA(cudaEventRecord(ev_s2e[s], streams[s]));

    // D2H: T and D
    CHECK_CUDA(cudaMemcpyAsync(hT_pin[s], d_T[s], needT, cudaMemcpyDeviceToHost, streams[s]));
    CHECK_CUDA(cudaMemcpyAsync(hD_pin[s], d_D[s], needD, cudaMemcpyDeviceToHost, streams[s]));

    // Finish this batch on this stream
    CHECK_CUDA(cudaStreamSynchronize(streams[s]));

    // Timing per batch (SpMMs)
    float ms1=0.f, ms2=0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms1, ev_s1s[s], ev_s1e[s]));
    CHECK_CUDA(cudaEventElapsedTime(&ms2, ev_s2s[s], ev_s2e[s]));
    report.spmm_ms += (ms1 + ms2);

    // Optional writes
    if (WRITE_BIN) {
      const std::string fT = "T_batch_" + std::to_string(b) + ".bin";
      const std::string fD = "D_batch_" + std::to_string(b) + ".bin";
      write_matrix_bin(fT, hT_pin[s], rows, nsub, rows, report.write_ms);
      write_matrix_bin(fD, hD_pin[s], rows, rows, rows, report.write_ms);
    }

    // Live progress
    processed_batches++;
    if (PROGRESS_EVERY > 0) {
      bool do_print = (processed_batches % PROGRESS_EVERY == 0) || (processed_batches == nonempty_batches);
      if (do_print) {
        double elapsed = now_ms() - t_start_ms;
        double per_batch = elapsed / processed_batches;
        double remain_ms = per_batch * (nonempty_batches - processed_batches);
        double eta_s = std::max(0.0, remain_ms / 1000.0);
        double bytesH2D = sizeof(int)*(rows+1) + sizeof(int)*nnz + sizeof(float)*nnz
                        + sizeof(float)*(size_t)nsub*nsub;
        double bytesD2H = sizeof(float)*(size_t)rows*nsub + sizeof(float)*(size_t)rows*rows;
        std::cout << "[progress] batch " << processed_batches << "/" << nonempty_batches
                  << " (stream " << s << ") rows=" << rows << " nnz=" << nnz
                  << " SpMM1="<<ms1<<"ms SpMM2="<<ms2<<"ms"
                  << " H2D="<< (bytesH2D/1024.0/1024.0) <<"MiB"
                  << " D2H="<< (bytesD2H/1024.0/1024.0) <<"MiB"
                  << " ETA="<< eta_s <<"s\n";
      }
    }

    // Destroy descriptors
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(Bm);
    cusparseDestroyDnMat(Tm);
    cusparseDestroyDnMat(Dm);
  }

  // Cleanup streams + buffers
  auto cl0=std::chrono::high_resolution_clock::now();
  for(int s=0;s<NSTREAMS;s++){
    CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    if(d_ro[s]) CHECK_CUDA(cudaFree(d_ro[s]));
    if(d_ci[s]) CHECK_CUDA(cudaFree(d_ci[s]));
    if(d_va[s]) CHECK_CUDA(cudaFree(d_va[s]));
    if(d_B[s])  CHECK_CUDA(cudaFree(d_B[s]));
    if(d_T[s])  CHECK_CUDA(cudaFree(d_T[s]));
    if(d_D[s])  CHECK_CUDA(cudaFree(d_D[s]));
    if(dWs[s])  CHECK_CUDA(cudaFree(dWs[s]));
    if(hT_pin[s]) CHECK_CUDA(cudaFreeHost(hT_pin[s]));
    if(hD_pin[s]) CHECK_CUDA(cudaFreeHost(hD_pin[s]));
    CHECK_CUDA(cudaEventDestroy(ev_s1s[s])); CHECK_CUDA(cudaEventDestroy(ev_s1e[s]));
    CHECK_CUDA(cudaEventDestroy(ev_s2s[s])); CHECK_CUDA(cudaEventDestroy(ev_s2e[s]));
    CHECK_CUDA(cudaStreamDestroy(streams[s]));
  }
  cusparseDestroy(cusH);
  report.cleanup_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-cl0).count();

  report.total_ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-T0).count();
  report.print();
  std::cout<<"Done. " << (WRITE_BIN? "Wrote T_batch_*.bin and D_batch_*.bin for non-empty batches.\n"
                                   : "File writing disabled (ABAT_WRITE=0).\n");
  return 0;
}
