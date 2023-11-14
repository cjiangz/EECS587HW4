#include <iostream>
#include <cstdio>      /* printf, fgets */
#include <cstdlib>     /* atoi */
#include <cmath>
#include <iomanip>

using namespace std;

__constant__ int n;

__device__ int getGlobalIdx_2D_1D(){
    int blockId = blockIdx.y + (blockIdx.x + 1) * gridDim.y;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int getGlobalId(int i, int j){
    return i*gridDim.y*blockDim.x + j;
}

__device__ bool isBoundary(){
    return (blockIdx.y == 0 && threadIdx.x == 0)
           || (blockIdx.y * blockDim.x + threadIdx.x >= n - 1);
}

__global__ void DoIter(double* A_old, double* A_new)
{
    __shared__ double block[1024];
    int myGlobalId = getGlobalIdx_2D_1D();
    int myLocalId = threadIdx.x;
    bool is_boundary = isBoundary();
    double temp = A_old[myGlobalId];
    block[myLocalId] = temp;
    __syncthreads();

    double above = A_old[myGlobalId - blockDim.x * gridDim.y];
    double below = A_old[myGlobalId + blockDim.x * gridDim.y];
    double left = myLocalId > 0? block[myLocalId - 1] : A_old[myGlobalId - 1];
    double right = myLocalId < 1023 ? block[myLocalId + 1] : A_old[myGlobalId + 1];
    double array[5] = {above,below,left,right,temp};

    for(int i=0;i < 3;++i) {
        int min_idx = i;
        for (int j = i; j < 5; ++j) {
            min_idx = array[min_idx] > array[j] ? j : min_idx;
        }
        double tmp = array[i];
        array[i] = array[min_idx];
        array[min_idx] = tmp;
    }

    if(!is_boundary) A_new[myGlobalId] = array[2];
}

__global__ void getVerificationValues(double* A, double* check1, double* check2){
    int myGlobalId = getGlobalIdx_2D_1D();
    int n_over_3_id = getGlobalId(n/3,n/3);
    int nineteen_thirtyseven_id = getGlobalId(19,37);
    if(myGlobalId == n_over_3_id){
        *check1 = A[myGlobalId];
    }
    if(myGlobalId == nineteen_thirtyseven_id){
        *check2 = A[myGlobalId];
    }
}

__global__ void reduce(double* A_old, double* A_new, int N) {
    __shared__ double block[1024];

    unsigned int myLocalId = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + myLocalId;
    block[myLocalId] = i >= N ? 0 : A_old[i];
    __syncthreads();

    for (unsigned int numElements=1024; numElements > 1; numElements /= 2) {
        if (myLocalId < numElements/2) {
            block[myLocalId] += block[myLocalId + numElements/2];
        }
        __syncthreads();
    }

    if (myLocalId == 0) A_new[blockIdx.x] = block[0];
}

int main(int argc, char** argv)
{
    int n_ = atoi(argv[1]);
    cudaMemcpyToSymbol(n,&n_,sizeof(n_));
    int rounded_n = n_;
    if(n_ % 1024 != 0){
        rounded_n = 1024*(n_/1024 + 1);
    }

    double *A = new double[n_* rounded_n];
    double *A_new = new double[n_ *rounded_n];
    double sumCheck;
    double check1;
    double check2;
    for(int i = 0; i < n_; ++i){
        for(int j=0;j< rounded_n;++j){
            if(j >= n_){
                A[i*rounded_n + j] = 0;
                continue;
            }
            A[i*rounded_n + j] = sin(i*i+j)*sin(i*i+j)+cos(i-j);
        }
    }

    dim3 gridDim(n_ - 2,rounded_n/1024,1);
    dim3 blockDim(1024,1,1);

    double* d_Aold;
    double* d_Anew;
    double *d_check1;
    double *d_check2;

    if(cudaMalloc(&d_Aold,sizeof(double)*n_*rounded_n) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_Anew,sizeof(double)*n_*rounded_n) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_check1,sizeof(double)) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_check2,sizeof(double)) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }

    if(cudaMemcpy(d_Aold,A,sizeof(double)*n_*rounded_n,cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy A into d_Aold"<<endl;
    }
    if(cudaMemcpy(d_Anew,A,sizeof(double)*n_*rounded_n,cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy A into d_Anew"<<endl;
    }

    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < 10; ++i){
        swap(d_Aold, d_Anew);
        DoIter<<<gridDim,blockDim>>>(d_Aold,d_Anew);
    }
    dim3 gridDim2(rounded_n,rounded_n/1024,1);
    dim3 blockDim2(1024,1,1);
    getVerificationValues<<<gridDim2,blockDim2>>>(d_Anew, d_check1, d_check2);

    int N = n_*rounded_n;
    while (N > 1) {
        swap(d_Aold, d_Anew);
        int rounded_N = N;
        if (N % 1024 != 0) {
            rounded_N = 1024 * (N / 1024 + 1);
        }
        reduce<<<rounded_N / 1024, 1024>>>(d_Aold, d_Anew, N);
        N = rounded_N/1024;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    if(cudaMemcpy(A_new,d_Anew,sizeof(double)*n_*rounded_n,cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_Anew into A_new"<<endl;
        cudaError_t error = cudaGetLastError();
        cout << cudaGetErrorString(error) << endl;
        cout << cudaGetErrorName(error) << endl;
    }

    if(cudaMemcpy(&check1, d_check1, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_check1 into check1"<<endl;
    }
    if(cudaMemcpy(&check2, d_check2, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_check2 into check2"<<endl;
    }
    if(cudaMemcpy(&sumCheck, d_Anew, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_Anew[0] into sumCheck"<<endl;
    }
    cout << std::setprecision(10) << "Sum: " << sumCheck <<  ",  A[n/3][n/3]:" << check1  << ", A[19][37]:" << check2 << ", Elapsed Time: " << elapsedTime << endl;

    delete[] A;
    delete[] A_new;
    cudaFree(d_Anew);
    cudaFree(d_Aold);
    cudaFree(d_check1);
    cudaFree(d_check2);

    return 0;
}