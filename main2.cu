#include <iostream>
#include <cstdio>      /* printf, fgets */
#include <cstdlib>     /* atoi */
#include <cmath>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

using namespace std;

__constant__ int n;

__device__ int getGlobalIdx_2D_1D(){
    int blockId = blockIdx.y + (blockIdx.x + 1) * gridDim.y;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

__device__ int getGlobalIdx_Above(){
    int block_x = blockIdx.x - 1;
    int thread_x = blockDim.x - 1;
    int block_y = blockIdx.y;
    int thread_y = threadIdx.y;
    int blockId = block_y + block_x * gridDim.y;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (thread_x * blockDim.y) + thread_y;
    return threadId;
}

__device__ int getGlobalIdx_Right(){

    int block_x = blockIdx.x;
    int thread_x = threadIdx.x;
    int block_y = blockIdx.y + 1;
    int thread_y = 0;
    int blockId = block_y + block_x * gridDim.y;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (thread_x * blockDim.y) + thread_y;
    return threadId;
}

__device__ int getGlobalIdx_Below(){
    int blockId = blockIdx.y + (blockIdx.x + 1) * gridDim.y;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (0 * blockDim.y) + threadIdx.y;
    return threadId;
}

__device__ int getGlobalIdx_Left(){
    int blockId = (blockIdx.y - 1) + blockIdx.x * gridDim.y;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.x * blockDim.y) + (blockDim.y - 1);
    return threadId;
}

__device__ int getGlobalId(int i, int j){
    return i*gridDim.y*blockDim.x + j;
}

__device__ int getLocalIdx_2D(){
    return threadIdx.x * blockDim.y + threadIdx.y;
}

__device__ bool isBoundary(){
    return (blockIdx.y == 0 && threadIdx.x == 0)
           || (blockIdx.y * blockDim.x + threadIdx.x >= n - 1);
}


__global__ void DoIter(double* A_old, double* A_new)
{
    //__shared__ double block[1024];
    int myGlobalId = getGlobalIdx_2D_1D();
    //int myLocalId = getLocalIdx_2D();
    bool is_boundary = isBoundary();
    //block[myLocalId] = A_old[myGlobalId];
    double temp = A_old[myGlobalId];
    //__syncthreads();

    double above = A_old[myGlobalId - blockDim.x * gridDim.y];
    double below = A_old[myGlobalId + blockDim.x * gridDim.y];
    double left = A_old[myGlobalId - 1];
    double right = A_old[myGlobalId + 1];
    double array[5] = {above,below,left,right,temp};
    // insertion sort
    /*
    for(int i=1;i < 5;++i){
        double tmp = array[i];
        int j = i - 1;
        for(;j >= 0 && array[j] > tmp; --j){
            array[j+1] = array[j];
        }
        array[j+1] = tmp;
    }
     */
    if(myGlobalId == 1025){
        //printf("lol %f %f %f %f %f %d\n", above, below, left, right, temp, is_boundary);
    }
    for(int i=0;i < 3;++i){
        int min_idx = i;
        for(int j = i;j < 5; ++j){
            min_idx = array[min_idx] > array[j] ? j : min_idx;
        }
        double tmp = array[i];
        array[i] = array[min_idx];
        array[min_idx] = tmp;
    }
    temp = array[2];


    if(!is_boundary) A_new[myGlobalId] = temp;
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
    __shared__ double sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = i >= N ? 0 : A_old[i];
    __syncthreads();

    for (unsigned int s=512; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) A_new[blockIdx.x] = sdata[0];
}


int getGlobalId(int i, int j, int rounded_n){
    int x = i/32;
    int x_id = i%32;
    int y = j/32;
    int y_id = j%32;
    int blockId = y + x * rounded_n/32;
    int globalId = blockId * (32 * 32)
                   + (x_id * 32) + y_id;
    return globalId;
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
        /*
        cudaDeviceSynchronize();
        if(cudaMemcpy(A_new,d_Anew,sizeof(double)*n_*rounded_n,cudaMemcpyDeviceToHost) != cudaSuccess){
            cout<<"Could not copy d_Anew into A_new"<<endl;
            cudaError_t error = cudaGetLastError();
            cout << cudaGetErrorString(error) << endl;
            cout << cudaGetErrorName(error) << endl;
        }
        for(int i = 0; i < n_; ++i){
            for(int j=0;j<rounded_n;++j){
                cout << A_new[i*rounded_n + j] << " ";
            }
            cout << endl;
        }
        cout << "finished" << endl << endl;
         */
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