//This is a cuda program that adds one integer
//It is overkill to use a GPU for this but shows the basics
//of initializing data on the host, copying it to the device
//launching a kernel, copying the result back, and freeing 
//the allocated memory


#include <iostream>

using namespace std;

__global__ void AddNum(int* a, int* b,int* c)
{
    //note how even though there is a scalar value we have to dereference
    //the variables, this is because when we used cudaMalloc it returns
    //a pointer. Yes, for those wondering since these pointers only have one value
    //this line could also be `*c = *a+*b`
    c[0] = a[0]+b[0];
    //*c = *a+*b;
}


//host function, note that the __host__ qualifier is not needed 
//as it is assumed by default
int main()
{
    //initialize host variables
    int A = 3, B = 12, C;

    //allocate memory for device variables
    int* d_a,*d_b,*d_c;


    //we can check if the cuda functions fail by seeing if they return a cudaSuccess code
    //you get status codes like cudaSuccess for free when you are compiling with nvcc
    if(cudaMalloc(&d_a,sizeof(int)) != cudaSuccess){
        cout<<"Could not allocate d_a"<<endl;
    }
    if(cudaMalloc(&d_b,sizeof(int)) != cudaSuccess){
        cout<<"Could not allocate d_b"<<endl;
    }
    if(cudaMalloc(&d_c,sizeof(int)) != cudaSuccess){
        cout<<"Could not allocate d_c"<<endl;
    }    

    //copy values of A and B to d_a and d_b

    //note that we provided d_a directly but had to provide &a, this is because
    //cudaMemcpy expects pointers
    if(cudaMemcpy(d_a,&A,sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy A into d_a"<<endl;
    }
    if(cudaMemcpy(d_b,&B,sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Could not copy B into d_b"<<endl;
    }
    
    AddNum<<<1,1>>>(d_a,d_b,d_c);
    
    //cudaMemcpy blocks until the kernel is finished and the data is copied, acting as a barrier
    //for our cout that prints the results, we could also call cudaDeviceSynchronize() above to
    //make the cpu wait until all kernels have finished
    //if we had any code above this cudaMemcpy call it would occur concurrently with the kernel
    //because the CPU can continue executing after the kernel is launched. This behavior could be
    //prevented by using a call to cudaDeviceSynchronize()
    //note that destination is first in the cudaMemcpy call
    if(cudaMemcpy(&C,d_c,sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Could not copy d_c into C"<<endl;
    }

    cout<<"The sum is: "<<C<<endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}