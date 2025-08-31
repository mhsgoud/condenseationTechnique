// combined_inverse_multiply.cu
// Compile: nvcc -O3 -std=c++17 -arch=native -o combined_inv_mult combined_inverse_multiply.cu -lcublas -lcudart -lcusparse
//
// Usage: ./combined_inv_mult sparse.txt indx_b.txt idx_std_loc.txt B1 B2
//
// B1 = number of batches in idx_b
// B2 = number of batches in idx_std_loc

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

// -------------------------------------------------------
// Error macros
// -------------------------------------------------------
#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

#define CHECK_CUBLAS(call) do {                               \
    cublasStatus_t s = (call);                                \
    if (s != CUBLAS_STATUS_SUCCESS) {                         \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",           \
                __FILE__, __LINE__, (int)s);                  \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

#define CHECK_CUSPARSE(call) do {                             \
    cusparseStatus_t status = (call);                         \
    if (status != CUSPARSE_STATUS_SUCCESS) {                  \
        fprintf(stderr, "cuSPARSE failed at line %d: %d\n",   \
                __LINE__, (int)status);                       \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// -------------------------------------------------------
// Sparse matrix reader (CSR format: m n nnz | row_ptr | cols | vals)
// -------------------------------------------------------
void read_sparse_matrix(const std::string &fname, int &m, int &n, int &nnz,
                        std::vector<int> &row_ptr, std::vector<int> &cols, std::vector<float> &vals)
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

// -------------------------------------------------------
// Batch indices reader (rows | cols)
// -------------------------------------------------------
void read_batches(const std::string &fname, int B,
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
        if(sep==std::string::npos){ std::cerr<<"Missing | in line\n"; exit(1);}
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
    if(b!=B) std::cerr<<"Warning: expected "<<B<<" batches, found "<<b<<"\n";
}

// -------------------------------------------------------
// Extract dense submatrix from CSR given row+col sets
// -------------------------------------------------------
std::vector<float> extract_dense_submatrix(
    const std::vector<int> &row_ptr,
    const std::vector<int> &cols,
    const std::vector<float> &vals,
    const std::vector<int> &row_indices,
    const std::vector<int> &col_indices)
{
    int nr=row_indices.size();
    int nc=col_indices.size();

    std::vector<float> dense(nr*nc,0.0f);
    std::vector<int> col_map( *std::max_element(col_indices.begin(),col_indices.end())+1, -1);
    for(int j=0;j<nc;j++) col_map[col_indices[j]]=j;

    for(int r=0;r<nr;r++){
        int i=row_indices[r];
        for(int jj=row_ptr[i]; jj<row_ptr[i+1]; jj++){
            int c=cols[jj];
            if(c>=0 && c<(int)col_map.size() && col_map[c]!=-1){
                int cc=col_map[c];
                dense[r*nc+cc]=vals[jj];
            }
        }
    }
    return dense;
}

// -------------------------------------------------------
// Print 4x4 block of a matrix
// -------------------------------------------------------
void print_4x4_block(const float* matrix, int n, int matrix_index)
{
    std::cout << "Matrix " << matrix_index << " (first 4x4 block):\n";
    for(int r = 0; r < std::min(4, n); r++) {
        for(int c = 0; c < std::min(4, n); c++) {
            std::cout << matrix[r * n + c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// -------------------------------------------------------
// Kernel for D = C * A^T
// -------------------------------------------------------
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
    int j = blockIdx.x + blockIdx.y * gridDim.x;
    if (j >= m) return;
    int tid = threadIdx.x;
    int tcount = blockDim.x;
    int start = A_rowptr[j];
    int end   = A_rowptr[j + 1];
    for (int row = tid; row < m; row += tcount) {
        float acc = 0.0f;
        for (int p = start; p < end; ++p) {
            int colk = A_cols[p];
            float aval = A_vals[p];
            acc += aval * C[row + colk * (size_t)m];
        }
        D[row + j * (size_t)m] = acc;
    }
}

// -------------------------------------------------------
// Main
// -------------------------------------------------------
int main(int argc,char **argv){
    if(argc<6){
        std::cerr<<"Usage: "<<argv[0]<<" sparse.txt indx_b.txt idx_std_loc.txt B1 B2\n";
        return 1;
    }
    std::string sparse_file=argv[1];
    std::string batch_file1=argv[2];
    std::string batch_file2=argv[3];
    int B1=std::stoi(argv[4]);
    int B2=std::stoi(argv[5]);

    // Read sparse matrix
    int m,n,nnz;
    std::vector<int> h_row_ptr,h_cols;
    std::vector<float> h_vals;
    read_sparse_matrix(sparse_file,m,n,nnz,h_row_ptr,h_cols,h_vals);

    // Read batch indices
    std::vector<std::vector<int>> row_batches1,col_batches1;
    std::vector<std::vector<int>> row_batches2,col_batches2;
    read_batches(batch_file1,B1,row_batches1,col_batches1);
    read_batches(batch_file2,B2,row_batches2,col_batches2);

    // Extract idx_b submatrices (to invert)
    std::vector<std::vector<float>> inv_matrices;
    int sub_n=-1;
    for(int b=0;b<B1;b++){
        auto mat = extract_dense_submatrix(h_row_ptr,h_cols,h_vals,row_batches1[b],col_batches1[b]);
        int nr = row_batches1[b].size();
        int nc = col_batches1[b].size();
        if(nr != nc){
            std::cerr<<"Error: idx_b submatrix not square ("<<nr<<"x"<<nc<<")\n";
            exit(1);
        }
        if(sub_n==-1) sub_n = nr;
        inv_matrices.push_back(std::move(mat));
    }

    // Extract idx_std_loc submatrices (for multiplication)
    std::vector<std::vector<float>> mult_matrices;
    for(int b=0;b<B2;b++){
        auto mat = extract_dense_submatrix(h_row_ptr,h_cols,h_vals,row_batches2[b],col_batches2[b]);
        mult_matrices.push_back(std::move(mat));
    }

    std::cout<<"Inversion matrices: "<<inv_matrices.size()<<", size "<<sub_n<<"x"<<sub_n<<"\n";
    std::cout<<"Multiplication matrices: "<<mult_matrices.size()<<"\n";

    // Pack inv_matrices into pinned host memory
    size_t matrix_elems = (size_t)sub_n*sub_n;
    size_t bytes = (size_t)inv_matrices.size()*matrix_elems*sizeof(float);
    float* h_A = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, bytes));
    for(size_t i=0;i<inv_matrices.size();i++)
        memcpy(h_A+i*matrix_elems, inv_matrices[i].data(), matrix_elems*sizeof(float));

    // Device allocations for inversion
    float* d_Ablock=nullptr;
    float* d_Cblock=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Ablock,bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Cblock,bytes));
    CHECK_CUDA(cudaMemcpy(d_Ablock,h_A,bytes,cudaMemcpyHostToDevice));

    // Pointer arrays for inversion
    std::vector<float*> h_Aarray(inv_matrices.size()), h_Carray(inv_matrices.size());
    for(size_t i=0;i<inv_matrices.size();i++){
        h_Aarray[i]=d_Ablock+i*matrix_elems;
        h_Carray[i]=d_Cblock+i*matrix_elems;
    }
    float** d_Aarray=nullptr;
    float** d_Carray=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Aarray,inv_matrices.size()*sizeof(float*)));
    CHECK_CUDA(cudaMalloc((void**)&d_Carray,inv_matrices.size()*sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_Aarray,h_Aarray.data(),inv_matrices.size()*sizeof(float*),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Carray,h_Carray.data(),inv_matrices.size()*sizeof(float*),cudaMemcpyHostToDevice));

    // cuBLAS handle for inversion
    cublasHandle_t blasHandle=nullptr;
    CHECK_CUBLAS(cublasCreate(&blasHandle));

    int* d_Pivots=nullptr;
    int* d_info=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Pivots,inv_matrices.size()*sub_n*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info,inv_matrices.size()*sizeof(int)));

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // LU factorization
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgetrfBatched(blasHandle,sub_n,d_Aarray,sub_n,d_Pivots,d_info,inv_matrices.size()));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout<<"LU factorization: "<<ms<<" ms\n";

    // Inversion
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgetriBatched(blasHandle,sub_n,(const float**)d_Aarray,sub_n,d_Pivots,d_Carray,sub_n,d_info,inv_matrices.size()));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout<<"Matrix inversion: "<<ms<<" ms\n";

    // Copy inverse results back
    float* h_inverses=nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_inverses,bytes));
    CHECK_CUDA(cudaMemcpy(h_inverses,d_Cblock,bytes,cudaMemcpyDeviceToHost));

    // Print inverse matrix snippets
    for(size_t b=0;b<inv_matrices.size();b++)
        print_4x4_block(h_inverses + b*matrix_elems, sub_n, b);

    // Multiplication D = C * A^T for each idx_std_loc batch
    for(size_t b=0;b<mult_matrices.size();b++){
        if(b >= inv_matrices.size()){
            std::cerr<<"Warning: Not enough inverse matrices for multiplication batch "<<b<<"\n";
            continue;
        }
        const auto& A = mult_matrices[b];
        int rows = row_batches2[b].size();
        int cols = col_batches2[b].size();

        // Extract sparse submatrix in CSR
        std::vector<int> csr_offsets(rows+1,0);
        std::vector<int> csr_cols;
        std::vector<float> csr_vals;
        std::vector<int> col_map(*std::max_element(col_batches2[b].begin(),col_batches2[b].end())+1,-1);
        for(int j=0;j<cols;j++) col_map[col_batches2[b][j]]=j;
        for(int r=0;r<rows;r++){
            int i = row_batches2[b][r];
            int count=0;
            for(int jj=h_row_ptr[i]; jj<h_row_ptr[i+1]; jj++){
                int c = h_cols[jj];
                if(c>=0 && c<(int)col_map.size() && col_map[c]!=-1){
                    csr_cols.push_back(col_map[c]);
                    csr_vals.push_back(h_vals[jj]);
                    count++;
                }
            }
            csr_offsets[r+1] = csr_offsets[r]+count;
        }
        int nnz = csr_vals.size();

        // Device memory
        int* d_csr_offsets=nullptr;
        int* d_csr_cols=nullptr;
        float* d_csr_vals=nullptr;
        float* d_C_dense=nullptr;
        float* dD=nullptr;
        CHECK_CUDA(cudaMalloc(&d_csr_offsets,(rows+1)*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csr_cols,nnz*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_csr_vals,nnz*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_dense,rows*cols*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dD,rows*rows*sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_csr_offsets,csr_offsets.data(),(rows+1)*sizeof(int),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csr_cols,csr_cols.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csr_vals,csr_vals.data(),nnz*sizeof(float),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C_dense,h_inverses+b*matrix_elems,rows*cols*sizeof(float),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dD,0,rows*rows*sizeof(float)));

        // Compute D = C * A^T using custom kernel
        int threads = 256;
        int maxGridX = 65535;
        int gx = std::min(rows,maxGridX);
        int gy = (rows+gx-1)/gx;
        dim3 grid(gx,gy);
        dim3 block(threads);
        csr_row_accumulate_columns_kernel<<<grid,block>>>(
            rows, cols,
            d_csr_offsets,
            d_csr_cols,
            d_csr_vals,
            d_C_dense,
            dD
        );
        CHECK_CUDA(cudaGetLastError());

        // Copy back and print
        std::vector<float> hD(rows*rows);
        CHECK_CUDA(cudaMemcpy(hD.data(),dD,rows*rows*sizeof(float),cudaMemcpyDeviceToHost));
        std::cout<<"Multiplication result batch "<<b<<" (first 4x4 block):\n";
        for(int r=0;r<std::min(4,rows);r++){
            for(int c=0;c<std::min(4,rows);c++)
                std::cout<<hD[r+c*(size_t)rows]<<" ";
            std::cout<<"\n";
        }
        std::cout<<"\n";

        // Cleanup
        CHECK_CUDA(cudaFree(d_csr_offsets));
        CHECK_CUDA(cudaFree(d_csr_cols));
        CHECK_CUDA(cudaFree(d_csr_vals));
        CHECK_CUDA(cudaFree(d_C_dense));
        CHECK_CUDA(cudaFree(dD));
    }

    // Final cleanup
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_inverses));
    CHECK_CUDA(cudaFree(d_Ablock));
    CHECK_CUDA(cudaFree(d_Cblock));
    CHECK_CUDA(cudaFree(d_Aarray));
    CHECK_CUDA(cudaFree(d_Carray));
    CHECK_CUDA(cudaFree(d_Pivots));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUBLAS(cublasDestroy(blasHandle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
