#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A = m x k 
// B = k x n
// C = m x n

#define m 1024
#define n 4096
#define k 2048
#define BLOCK_SIZE 16

__global__ void Simple_MatMul(float *_a, float *_b, float *_c, int m, int n, int k) {
    
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    
    if( row >= m || col >= n)
        return;
    
    float ret = 0;
    __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
    int localRow = threadIdx.x;
    int localCol = threadIdx.y;


    for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
        int block_offset = bID * BLOCK_SIZE;

        if(row >= m || block_offset + localCol >= k)
            subA[localRow][localCol] = 0;
        else
            subA[localRow][localCol] = _a[(row * k) + block_offset + localCol];
        
        if(col >= n || block_offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = _b[(block_offset + localRow)*n + col];
        
        __syncthreads();

        for(int i = 0; i < BLOCK_SIZE; i++) {
            ret += __fmul_rn(subA[localRow][i], subB[i][localCol]);
        }

        __syncthreads();
    }

    _c[(n * row + col)] = ret;

}

int main(int argc, int** argv) {

    int size_A = m * k;
    int size_B = k * n;
    int size_C = m * n;

    float *a, *b, *c;
    float *_a, *_b, *_c;

    a = new float[size_A]; memset(a,0,size_A);
    b = new float[size_B]; memset(b,0,size_B);
    c = new float[size_C]; memset(c,0,size_C);

    for (int i = 0; i < size_A; i++) {
        a[i] = rand() % 10 + ((rand() % 100) / 100.0);
    }
    for (int i = 0; i < size_B; i++) {
        b[i] = rand() % 10 + ((rand() % 100) / 100.0);
    }

    cudaMalloc(&_a, size_A);
    cudaMalloc(&_b, size_B);
    cudaMalloc(&_c, size_C);

    cudaMemcpy(_a, a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size_B, cudaMemcpyHostToDevice);

    dim3 gridDim (ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim (BLOCK_SIZE, BLOCK_SIZE);

    Simple_MatMul<<<gridDim, blockDim>>>(_a,_b,_c,m,n,k);

    cudaMemcpy(c, _c, size_C, cudaMemcpyDeviceToHost);

    bool result = true;
    for (int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float ret = 0;
            for(int l = 0; l < k; l++)
                ret += a[(k*i)+l] * b[(l*n) + j];
            if(ret != c[i*n + j]) {
                printf("the result is not matched! (%d, %d)\n"
                , i, ret, c[i*n + j]);
                result = false;
            }
        }
    }
    if(result)
        printf("kernel works well!\n");
    return 0;
}