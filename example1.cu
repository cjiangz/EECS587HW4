//This is a cuda program that adds one integer
//It is overkill to use a GPU for this but shows the basics
//of initializing data on the host, copying it to the device
//launching a kernel, copying the result back, and freeing 
//the allocated memory


#include <iostream>
#include <cstdio>      /* printf, fgets */
#include <cstdlib>     /* atoi */
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>
#include <numeric>


using namespace std;

__constant__ int n;

__device__ uint32_t getGlobalIdx_2D_2D(){
    uint32_t blockId = blockIdx.y + blockIdx.x * gridDim.y;
    uint32_t threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.x * blockDim.y) + threadIdx.y;
    return threadId;
}

__device__ uint32_t getLocalIdx_2D(){
    return threadIdx.x * blockDim.y + threadIdx.y;
}

__device__ bool isBoundary(){
    return (blockIdx.x == 0 && threadIdx.x == 0)
    || (blockIdx.x * blockDim.x + threadIdx.x >= n - 1)
    || (blockIdx.y == 0 && threadIdx.y == 0)
    || (blockIdx.y * blockDim.y + threadIdx.y >= n - 1);
}

__global__ void DoIter(double* A_old, double* A_new)
{
    __shared__ double block[1024];
    uint32_t myGlobalId = getGlobalIdx_2D_2D();
    uint32_t myLocalId = getLocalIdx_2D();
    bool is_boundary = isBoundary();
    block[myLocalId] = A_old[myGlobalId];
    double temp = block[threadIdx.x];
    __syncthreads();
    if(!is_boundary){
        double above = threadIdx.x == 0 ? A_old[myGlobalId - gridDim.y * blockDim.y] : block[myLocalId - blockDim.y]; //high probability
        double below = threadIdx.x == blockDim.x ? A_old[myGlobalId + gridDim.y * blockDim.y] : block[myLocalId + blockDim.y];
        double left = threadIdx.y == 0 ? A_old[myGlobalId - 1] : block[myLocalId - 1];
        double right = threadIdx.y == blockDim.y - 1 ? A_old[myGlobalId + 1] : block[myLocalId + 1];
        double array[5] = {above,below,left,right,temp};
        // insertion sort
        for(int i=1;i < 5;++i){
            double tmp = array[i];
            int j = i - 1;
            for(;j >= 0 && array[j] > tmp; --j){
                array[j+1] = array[j];
            }
            array[j+1] = tmp;
        }
        temp = array[2];
    }
    A_new[myGlobalId] = temp;
}

__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



//host function, note that the __host__ qualifier is not needed 
//as it is assumed by default
int main(int argc, char** argv)
{
    //initialize host variables
    int n_ = atoi(argv[1]);
    cudaMemcpyToSymbol(n,&n_,sizeof(n_));
    int rounded_n = n;
    if(n % 32 != 0){
        rounded_n = 32*(n/32 + 1);
    }
    vector<double> A(rounded_n * rounded_n,0);
    for(size_t i = 0; i < n; ++i){
        for(size_t j=0;j<n;++j){
            A[i*rounded_n + j] = sin(i*i+j)*sin(i*i+j)+cos(i-j);
        }
    }

    dim3 gridDim(rounded_n/32,rounded_n/32,1);
    dim3 blockDim(32,32,1);

    //allocate memory for device variables
    double* d_Aold;
    double* d_Anew;
    double* d_check1;
    double *d_check2;
    double *d_sumCheck;

    //we can check if the cuda functions fail by seeing if they return a cudaSuccess code
    //you get status codes like cudaSuccess for free when you are compiling with nvcc
    if(cudaMalloc(&d_Aold,sizeof(double)*rounded_n*rounded_n) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_Anew,sizeof(double)*rounded_n*rounded_n) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_check1,sizeof(double)) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_check2,sizeof(double)) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }
    if(cudaMalloc(&d_sumCheck,sizeof(double)) != cudaSuccess){
        cout<<"Could not allocate d_A"<<endl;
    }

    //copy values of A and B to d_a and d_b

    //note that we provided d_a directly but had to provide &a, this is because
    //cudaMemcpy expects pointers
    if(cudaMemcpy(d_Aold,A.data(),sizeof(double)*rounded_n*rounded_n,cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy A into d_Aold"<<endl;
    }
    if(cudaMemcpy(d_Anew,A.data(),sizeof(double)*rounded_n*rounded_n,cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy A into d_Anew"<<endl;
    }
    for(int i = 0; i < 10; ++i){
        swap(d_Aold, d_Anew);
        DoIter<<<gridDim,blockDim>>>(d_Aold,d_Anew);
        cudaDeviceSynchronize();
    }

    
    //AddNum<<<1,1>>>(d_a,d_b,d_c);
    
    //cudaMemcpy blocks until the kernel is finished and the data is copied, acting as a barrier
    //for our cout that prints the results, we could also call cudaDeviceSynchronize() above to
    //make the cpu wait until all kernels have finished
    //if we had any code above this cudaMemcpy call it would occur concurrently with the kernel
    //because the CPU can continue executing after the kernel is launched. This behavior could be
    //prevented by using a call to cudaDeviceSynchronize()
    //note that destination is first in the cudaMemcpy call
    if(cudaMemcpy(A.data(),d_Anew,sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_Anew into A"<<endl;
    }

    cout << std::accumulate(A.begin(), A.end(), 0.0) << " " << A[19*rounded_n + 37] << " "
        << A[(n/3)* rounded_n + (n/3)] << endl;

    cudaFree(d_Anew);
    cudaFree(d_Aold);
    cudaFree(d_sumCheck);
    cudaFree(d_check1);
    cudaFree(d_check2);

    return 0;
}