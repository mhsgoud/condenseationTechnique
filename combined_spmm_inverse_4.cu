// combined_spmm_inverse.cu
// Compile: nvcc -O3 -std=c++17 -arch=native -o combined_spmm_inverse combined_spmm_inverse.cu -lcublas -lcudart -lcusparse
//
// Usage:
//   ./combined_spmm_inverse sparse.txt indx_b.txt idx_std_loc.txt B1 B2
//
// B1 = number of batches in indx_b.txt (these will be inverted)
// B2 = number of batches in idx_std_loc.txt (these will be used as sparse A batches)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t s = (call); \
    if (s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel from your SpMM+accumulate code to compute D = C * A^T
__global__
void csr_row_accumulate_columns_kernel(
    int m,                    // rows in A, rows in C (and number of rows in D)
    int n,                    // cols in A, cols in C
    const int* __restrict__ A_rowptr,
    const int* __restrict__ A_cols,
    const float* __restrict__ A_vals,
    const float* __restrict__ C, // dense m x n (column-major, ld = m)
    float* __restrict__ D       // dense m x m (column-major, ld = m); D(:,j) computed here
) {
    // compute column j as a 2D-grid flatten
    int j = blockIdx.x + blockIdx.y * gridDim.x;
    if (j >= m) return;

    int tid = threadIdx.x;
    int tcount = blockDim.x;

    int start = A_rowptr[j];
    int end   = A_rowptr[j + 1];

    for (int row = tid; row < m; row += tcount) {
        float acc = 0.0f;
        for (int p = start; p < end; ++p) {
            int colk = A_cols[p];       // column index k
            float aval = A_vals[p];     // A(j,k)
            // C is m x n in column-major, column colk starts at C + colk * m
            acc += aval * C[row + (size_t)colk * m];
        }
        D[row + (size_t)j * m] = acc;
    }
}

// ---------------- IO and helpers ----------------
void read_sparse_matrix(const std::string &fname,int &m,int &n,int &nnz,
                        std::vector<int> &row_ptr,std::vector<int> &cols,std::vector<float> &vals)
{
    std::ifstream fin(fname);
    if(!fin){ std::cerr<<"Cannot open "<<fname<<"\n"; exit(1);}
    fin>>m>>n>>nnz;
    row_ptr.resize(m+1);
    for(int i=0;i<=m;i++) fin>>row_ptr[i];
    cols.resize(nnz);
    for(int i=0;i<nnz;i++) fin>>cols[i];
    vals.resize(nnz);
    for(int i=0;i<nnz;i++) fin>>vals[i];
}

void read_batches_file(const std::string &fname,int B,
                  std::vector<std::vector<int>> &row_batches,
                  std::vector<std::vector<int>> &col_batches)
{
    row_batches.clear();
    col_batches.clear();
    std::ifstream fin(fname);
    if(!fin){ std::cerr<<"Cannot open "<<fname<<"\n"; exit(1);}
    std::string line;
    int b=0;
    while(std::getline(fin,line)){
        if(line.empty()) continue;
        size_t sep=line.find('|');
        if(sep==std::string::npos){ std::cerr<<"Missing | in line in "<<fname<<"\n"; exit(1);}
        std::string left=line.substr(0,sep);
        std::string right=line.substr(sep+1);
        std::stringstream lss(left),rss(right);
        std::vector<int> rows,cols;
        int v;
        while(lss>>v) rows.push_back(v);
        while(rss>>v) cols.push_back(v);
        row_batches.push_back(rows);
        col_batches.push_back(cols);
        b++;
    }
    if(b!=B) std::cerr<<"Warning: expected "<<B<<" batches in "<<fname<<", found "<<b<<"\n";
}

// Extract dense submatrix but **column-major** layout (so it's ready for cuBLAS/cuSPARSE)
std::vector<float> extract_dense_submatrix_colmajor(
    const std::vector<int> &row_ptr,
    const std::vector<int> &cols,
    const std::vector<float> &vals,
    const std::vector<int> &row_indices,
    const std::vector<int> &col_indices)
{
    int nr = (int)row_indices.size();
    int nc = (int)col_indices.size();
    if(nr!=nc){
        std::cerr<<"Submatrix not square ("<<nr<<"x"<<nc<<")\n";
        exit(1);
    }

    int max_col = nc? *std::max_element(col_indices.begin(), col_indices.end()) : -1;
    std::vector<int> col_map(max_col+1, -1);
    for(int j=0;j<nc;j++){
        col_map[col_indices[j]] = j;
    }

    std::vector<float> dense((size_t)nr*nc, 0.0f);
    // store column-major: element (r,c) -> dense[c*nr + r]
    for(int r=0;r<nr;r++){
        int i = row_indices[r];
        for(int jj = row_ptr[i]; jj < row_ptr[i+1]; ++jj){
            int c = cols[jj];
            if(c>=0 && c <= max_col){
                int cc = col_map[c];
                if(cc!=-1) dense[(size_t)cc*nr + r] = vals[jj];
            }
        }
    }
    return dense;
}

void write_matrix_to_file(const std::string &fname, const float* data, int rows, int cols) {
    std::ofstream fout(fname);
    if(!fout) {
        std::cerr << "Cannot open file " << fname << " for writing\n";
        return;
    }
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            fout << data[c*(size_t)rows + r] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

template<typename T>
void write_vector_to_file(const std::string &fname, const std::vector<T> &vec) {
    std::ofstream fout(fname);
    if(!fout) {
        std::cerr << "Cannot open file " << fname << " for writing\n";
        return;
    }
    for(auto &v : vec) fout << v << " ";
    fout << "\n";
    fout.close();
}

int main(int argc, char** argv){
    if(argc < 6){
        std::cerr<<"Usage: "<<argv[0]<<" sparse.txt indx_b.txt idx_std_loc.txt B1 B2\n";
        return 1;
    }
    std::string sparse_file = argv[1];
    std::string indx_b_file  = argv[2]; // we will invert these
    std::string idx_std_file = argv[3]; // sparse A used for SpMM
    int B1 = std::stoi(argv[4]); // number of batches in indx_b (to invert)
    int B2 = std::stoi(argv[5]); // number of batches in idx_std_loc (sparse A batches)

    // Read full sparse matrix (CSR) used for extraction
    int m_global,n_global,nnz_global;
    std::vector<int> h_row_ptr, h_cols;
    std::vector<float> h_vals;
    read_sparse_matrix(sparse_file, m_global, n_global, nnz_global, h_row_ptr, h_cols, h_vals);
    std::cout<<"Read sparse matrix m="<<m_global<<", n="<<n_global<<", nnz="<<nnz_global<<"\n";

    // Read batches: indx_b (to invert) and idx_std_loc (sparse A to use)
    std::vector<std::vector<int>> rows_b1, cols_b1;
    std::vector<std::vector<int>> rows_b2, cols_b2;
    read_batches_file(indx_b_file, B1, rows_b1, cols_b1);
    read_batches_file(idx_std_file, B2, rows_b2, cols_b2);

    // -------------------------
    // Extract dense matrices for B1 (to be inverted). store column-major
    // -------------------------
    std::vector<std::vector<float>> dense_list; dense_list.reserve(B1);
    int nsub = -1;
    for(int b=0;b<B1;b++){
        if(rows_b1[b].size() != cols_b1[b].size()){
            std::cerr<<"Error: indx_b batch "<<b<<" is not square\n"; return 1;
        }
        if(nsub==-1) nsub = (int)rows_b1[b].size();
        else if((int)rows_b1[b].size() != nsub){
            std::cerr<<"Error: indx_b batch "<<b<<" has different size than previous\n"; return 1;
        }
        auto dense = extract_dense_submatrix_colmajor(h_row_ptr, h_cols, h_vals, rows_b1[b], cols_b1[b]);
        dense_list.push_back(std::move(dense));
    }
    if(nsub<=0){ std::cerr<<"No submatrices to invert\n"; return 1; }

    std::cout<<"Extracted "<<dense_list.size()<<" dense submatrices of size "<<nsub<<"x"<<nsub<<"\n";

    // -------------------------
    // Invert dense_list using cuBLAS batched routines
    // We will pack them into pinned host memory in column-major order (already col-major)
    // -------------------------
    int batchSize = (int)dense_list.size();
    size_t matrix_elems = (size_t)nsub * nsub;
    size_t total_elems = matrix_elems * batchSize;
    size_t bytes = total_elems * sizeof(float);

    float* h_A = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, bytes)); // pinned
    for(int b=0;b<batchSize;b++){
        memcpy(h_A + (size_t)b*matrix_elems, dense_list[b].data(), matrix_elems*sizeof(float));
    }

    // device allocations for inversion
    float* d_Ablock = nullptr;
    float* d_Cblock = nullptr; // will hold inverses (column-major)
    CHECK_CUDA(cudaMalloc((void**)&d_Ablock, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Cblock, bytes));
    CHECK_CUDA(cudaMemcpy(d_Ablock, h_A, bytes, cudaMemcpyHostToDevice));

    // create pointer arrays for batched cuBLAS (device pointers)
    std::vector<float*> h_Aarray(batchSize), h_Carray(batchSize);
    for(int i=0;i<batchSize;i++){
        h_Aarray[i] = d_Ablock + (size_t)i * matrix_elems;
        h_Carray[i] = d_Cblock + (size_t)i * matrix_elems;
    }
    float** d_Aarray = nullptr;
    float** d_Carray = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Aarray, batchSize * sizeof(float*)));
    CHECK_CUDA(cudaMalloc((void**)&d_Carray, batchSize * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_Aarray, h_Aarray.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Carray, h_Carray.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t cublasHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // pivot and info arrays
    int* d_Pivots = nullptr;
    int* d_info = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Pivots, batchSize * nsub * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, batchSize * sizeof(int)));

    // events & run LU + inverse
    cudaEvent_t e0, e1; CHECK_CUDA(cudaEventCreate(&e0)); CHECK_CUDA(cudaEventCreate(&e1));
    float ms=0.f;

    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUBLAS(cublasSgetrfBatched(cublasHandle, nsub, d_Aarray, nsub, d_Pivots, d_info, batchSize));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1)); CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    std::cout<<"getrfBatched: "<<ms<<" ms\n";

    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUBLAS(cublasSgetriBatched(cublasHandle, nsub, (const float**)d_Aarray, nsub, d_Pivots, d_Carray, nsub, d_info, batchSize));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1)); CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    std::cout<<"getriBatched: "<<ms<<" ms\n";

    // Copy inverses back to host pinned if you want (optional). We'll keep device pointer d_Cblock.
    float* h_C = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_C, bytes));
    CHECK_CUDA(cudaMemcpy(h_C, d_Cblock, bytes, cudaMemcpyDeviceToHost));
    std::cout<<"Inversion complete. First inverse (5x5 snippet, column-major):\n";
    for(int r=0;r<std::min(5,nsub);r++){
        for(int c=0;c<std::min(5,nsub);c++){
            // column-major layout: (r,c) at h_C[c*nsub + r]
            printf("%9.6f ", h_C[c*(size_t)nsub + r]);
        }
        printf("\n");
    }

    for (int b = 0; b < B1; b++)
    {
        std::string fname = "out/inverse_batch_" + std::to_string(b) + ".txt";
        write_matrix_to_file(fname, h_C + (size_t)b * matrix_elems, nsub, nsub);
    }
    std::cout << "Inverted matrices written to files.\n";

    // -------------------------
    // Now process idx_std_loc batches: treat each A (sparse) and use corresponding inverse as B.
    // Requirements: B_num_cols == nsub (square inverse). Also B2 <= B1.
    // -------------------------
    if(B2 > B1){
        std::cerr<<"Error: number of sparse batches (B2="<<B2<<") is greater than inverted batches (B1="<<B1<<").\n";
        return 1;
    }

    // Read idx_std_loc batches and prepare CSR and placeholders
    // We'll build device arrays similar to your earlier SpMM program.
    // For simplicity, read the idx_std_file batches using the same read_batches_file (they're already read into rows_b2,cols_b2).
    // But we need per-batch CSR arrays extracted from the global CSR (like your extractor).
    std::vector<int> A_num_rows(B2), A_num_cols(B2), A_nnz(B2);
    std::vector<std::vector<int>> hA_csrOffsets(B2), hA_columns(B2);
    std::vector<std::vector<float>> hA_values(B2);

    for(int b=0;b<B2;b++){
        int rows = (int)rows_b2[b].size();
        int cols = (int)cols_b2[b].size();
        // Build mapping from global col index to local col index (we assume cols_b2[b] contains column indices)
        // We'll produce CSR for the submatrix that picks rows rows_b2[b] and columns cols_b2[b].
        // This builds CSR by scanning each row and keeping entries whose col is in cols_b2[b].
        int max_col = cols? *std::max_element(cols_b2[b].begin(), cols_b2[b].end()) : -1;
        std::vector<int> col_map(max_col+1, -1);
        for(int j=0;j<cols;j++) col_map[cols_b2[b][j]] = j;
        // Build CSR
        hA_csrOffsets[b].resize(rows+1);
        hA_columns[b].clear();
        hA_values[b].clear();
        int accum = 0;
        for(int ri=0; ri<rows; ++ri){
            int global_row = rows_b2[b][ri];
            hA_csrOffsets[b][ri] = accum;
            for(int jj = h_row_ptr[global_row]; jj < h_row_ptr[global_row+1]; ++jj){
                int gc = h_cols[jj];
                if(gc >= 0 && gc <= max_col && col_map[gc] != -1){
                    hA_columns[b].push_back(col_map[gc]);
                    hA_values[b].push_back(h_vals[jj]);
                    accum++;
                }
            }
        }
        hA_csrOffsets[b][rows] = accum;
        A_num_rows[b] = rows;
        A_num_cols[b] = cols;
        A_nnz[b] = accum;
        std::cout<<"Prepared sparse batch "<<b<<": rows="<<rows<<", cols="<<cols<<", nnz="<<accum<<"\n";
        // size check: cols must equal nsub so A*B is valid where B is nsub x nsub
        if(cols != nsub){
            std::cerr<<"Error: idx_std_loc batch "<<b<<" has cols="<<cols<<" but inverted matrices are "<<nsub<<".\n";
            return 1;
        }
    }

    for (int b = 0; b < B2; b++)
    {
        write_vector_to_file("out/batch_" + std::to_string(b) + "_values.txt", hA_values[b]);
        write_vector_to_file("out/batch_" + std::to_string(b) + "_columns.txt", hA_columns[b]);
        write_vector_to_file("out/batch_" + std::to_string(b) + "_rowptr.txt", hA_csrOffsets[b]);
    }
    std::cout << "CSR batch arrays written to files.\n";

   



// ... (previous code remains the same until after the inversion part)

    // After inversion, we need to write the output in the specified format
    std::ofstream fout("output_result.txt");
    if (!fout) {
        std::cerr << "Cannot open output_result.txt for writing\n";
        return 1;
    }

    // Write number of sets (batches)
    fout << B2 << "\n";

    // First, write the dimensions for each sparse matrix
    for (int b = 0; b < B2; b++) {
        fout << A_num_rows[b] << " " << A_num_cols[b] << " " << A_nnz[b] << "\n";
    }

    // Write the number of columns for the dense matrices (all are nsub x nsub)
    fout << nsub << "\n";

    // Now write each batch: sparse matrix CSR + dense matrix
    for (int b = 0; b < B2; b++) {
        // Write sparse matrix row pointers (0-based)
        for (size_t i = 0; i < hA_csrOffsets[b].size(); i++) {
            fout << hA_csrOffsets[b][i];
            if (i < hA_csrOffsets[b].size() - 1) fout << " ";
        }
        fout << "\n";

        // Write sparse matrix column indices (0-based)
        for (size_t i = 0; i < hA_columns[b].size(); i++) {
            fout << hA_columns[b][i];
            if (i < hA_columns[b].size() - 1) fout << " ";
        }
        fout << "\n";

        // Write sparse matrix values
        for (size_t i = 0; i < hA_values[b].size(); i++) {
            fout << hA_values[b][i];
            if (i < hA_values[b].size() - 1) fout << " ";
        }
        fout << "\n";

        // Write dense matrix (inverse) in column-major order
        // Get the inverse matrix for this batch
        const float* inverse_matrix = h_C + (size_t)b * matrix_elems;
        for (int col = 0; col < nsub; col++) {
            for (int row = 0; row < nsub; row++) {
                fout << inverse_matrix[col * (size_t)nsub + row];
                if (!(col == nsub - 1 && row == nsub - 1)) fout << " ";
            }
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "Output written to output_result.txt in the specified format\n";

    // free inversion allocations
    // CHECK_CUDA(cudaFree(d_Ablock));
    // ... (rest of the cleanup code remains the same)






    // free inversion allocations
    CHECK_CUDA(cudaFree(d_Ablock));
    CHECK_CUDA(cudaFree(d_Cblock));
    CHECK_CUDA(cudaFree(d_Aarray));
    CHECK_CUDA(cudaFree(d_Carray));
    CHECK_CUDA(cudaFree(d_Pivots));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_C));

    cublasDestroy(cublasHandle);

    std::cout<<"Done.\n";
    return 0;
}
