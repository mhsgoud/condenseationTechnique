// combined_inverse.cu
// Compile: nvcc -O3 -std=c++17 -arch=native -o combined_inv combined_inverse.cu -lcublas -lcudart
//
// Usage: ./combined_inv sparse.txt indx_b.txt idx_std_loc.txt B1 B2
//
// B1 = number of batches in indx_b
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

// -------------------------------------------------------
// Sparse matrix reader (CSR format: m n nnz | row_ptr | cols | vals)
// -------------------------------------------------------
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

// -------------------------------------------------------
// Batch indices reader (rows | cols)
// -------------------------------------------------------
void read_batches(const std::string &fname,int B,
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
    if(nr!=nc){
        std::cerr<<"Submatrix not square ("<<nr<<"x"<<nc<<")\n";
        exit(1);
    }

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

    int m,n,nnz;
    std::vector<int> h_row_ptr,h_cols;
    std::vector<float> h_vals;
    read_sparse_matrix(sparse_file,m,n,nnz,h_row_ptr,h_cols,h_vals);

    std::vector<std::vector<int>> row_batches1,col_batches1;
    std::vector<std::vector<int>> row_batches2,col_batches2;
    read_batches(batch_file1,B1,row_batches1,col_batches1);
    read_batches(batch_file2,B2,row_batches2,col_batches2);

    // Extract all submatrices into dense
    std::vector<std::vector<float>> submatrices;
    int sub_n=-1;
    for(int b=0;b<B1;b++){
        auto mat=extract_dense_submatrix(h_row_ptr,h_cols,h_vals,row_batches1[b],col_batches1[b]);
        if(sub_n==-1) sub_n=row_batches1[b].size();
        submatrices.push_back(std::move(mat));
    }
    for(int b=0;b<B2;b++){
        auto mat=extract_dense_submatrix(h_row_ptr,h_cols,h_vals,row_batches2[b],col_batches2[b]);
        submatrices.push_back(std::move(mat));
    }

    int batchSize=submatrices.size();
    int nsub=sub_n; // square size
    std::cout<<"Total submatrices: "<<batchSize<<", size "<<nsub<<"x"<<nsub<<"\n";

    // Pack into pinned host memory
    size_t matrix_elems=(size_t)nsub*nsub;
    size_t total_elems=(size_t)batchSize*matrix_elems;
    size_t bytes=total_elems*sizeof(float);

    float* h_A=nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_A,bytes));
    for(int b=0;b<batchSize;b++){
        memcpy(h_A+b*matrix_elems,submatrices[b].data(),matrix_elems*sizeof(float));
    }

    // Device allocations
    float* d_Ablock=nullptr;
    float* d_Cblock=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Ablock,bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Cblock,bytes));
    CHECK_CUDA(cudaMemcpy(d_Ablock,h_A,bytes,cudaMemcpyHostToDevice));

    // Pointer arrays
    std::vector<float*> h_Aarray(batchSize),h_Carray(batchSize);
    for(int i=0;i<batchSize;i++){
        h_Aarray[i]=d_Ablock+(size_t)i*matrix_elems;
        h_Carray[i]=d_Cblock+(size_t)i*matrix_elems;
    }
    float** d_Aarray=nullptr;
    float** d_Carray=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Aarray,batchSize*sizeof(float*)));
    CHECK_CUDA(cudaMalloc((void**)&d_Carray,batchSize*sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_Aarray,h_Aarray.data(),batchSize*sizeof(float*),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Carray,h_Carray.data(),batchSize*sizeof(float*),cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle=nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));

    int* d_Pivots=nullptr;
    int* d_info=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Pivots,batchSize*nsub*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info,batchSize*sizeof(int)));

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // LU factorization
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgetrfBatched(handle,nsub,d_Aarray,nsub,d_Pivots,d_info,batchSize));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout<<"LU factorization: "<<ms<<" ms\n";

    // Inversion
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgetriBatched(handle,nsub,(const float**)d_Aarray,nsub,d_Pivots,
                                     d_Carray,nsub,d_info,batchSize));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout<<"Matrix inversion: "<<ms<<" ms\n";

    // Copy results back
    float* h_C=nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_C,bytes));
    CHECK_CUDA(cudaMemcpy(h_C,d_Cblock,bytes,cudaMemcpyDeviceToHost));

    // Print first matrix inverse snippet
    std::cout<<"Inverse[0] (first 5x5 block):\n";
    for(int r=0;r<std::min(5,nsub);r++){
        for(int c=0;c<std::min(5,nsub);c++){
            std::cout<<h_C[r*nsub+c]<<" ";
        }
        std::cout<<"\n";
    }

    // cleanup
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_C));
    CHECK_CUDA(cudaFree(d_Ablock));
    CHECK_CUDA(cudaFree(d_Cblock));
    CHECK_CUDA(cudaFree(d_Aarray));
    CHECK_CUDA(cudaFree(d_Carray));
    CHECK_CUDA(cudaFree(d_Pivots));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
