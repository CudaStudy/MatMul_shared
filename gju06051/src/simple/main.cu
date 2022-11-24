#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW_SIZE (32)
#define K_SIZE (128)
#define COL_SIZE (32)

#define WORK_LOAD (1024)

#define CPU 0
#define GPU 1
#define CPU2GPU 2
#define GPU2CPU 3
#define NSH_GPU 4
#define NSH_CPU2GPU 5
#define NSH_GPU2CPU 6

__global__ void matMul_kernel_shared(float *_A, float *_B, float *_C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int index = row * blockDim.x + col;

    __shared__ float sA[ROW_SIZE][K_SIZE]; // 32 * 256 * 4 bytes = 16 KB
    __shared__ float sB[K_SIZE][COL_SIZE]; // 16 KB

    int offset = 0;

    // load A
    int numSubMatA = ceil((float)K_SIZE / COL_SIZE);
    for (int i = 0; i < numSubMatA; i++)
    {
        if (col + offset >= K_SIZE)
            break;

        sA[row][col + offset] = _A[row * K_SIZE + (col + offset)];
        offset += COL_SIZE;
    }

    // load B
    offset = 0;
    int numSubMatB = ceil((float)K_SIZE / ROW_SIZE);
    for (int i = 0; i < numSubMatB; i++)
    {
        if (row + offset >= K_SIZE)
            break;

        sB[row + offset][col] = _B[col + (row + offset) * COL_SIZE];
        offset += ROW_SIZE;
    }

    __syncthreads(); // wait until all thread load the matrix

    _C[index] = 0;
    for (int k = 0; k < K_SIZE; k++)
        for (int i = 0; i < WORK_LOAD; i++)
            _C[index] += sA[row][k] * sB[k][col];
}
__global__ void matMul_kernel(float *_A, float *_B, float *_C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int index = row * blockDim.x + col;

    _C[index] = 0;
    for (int k = 0; k < K_SIZE; k++)
        for (int i = 0; i < WORK_LOAD; i++)
            _C[index] += __fmul_rn(_A[row * K_SIZE + k], _B[col + k * COL_SIZE]);
}

int main(void)
{
    DS_timer timer(7);
    timer.setTimerName(CPU, (char *)"[CPU]");
    timer.setTimerName(GPU, (char *)"[GPU_SHARED] : Multilplication");
    timer.setTimerName(CPU2GPU, (char *)"[GPU_SHARED] : Host->Device");
    timer.setTimerName(GPU2CPU, (char *)"[GPU_SHARED] : Device->Host");
    timer.setTimerName(NSH_GPU, (char *)"[GPU_NOT_SHARED] : Multiplication");
    timer.setTimerName(NSH_CPU2GPU, (char *)"[GPU_NOT_SHARED] : Host->Device");
    timer.setTimerName(NSH_GPU2CPU, (char *)"[GPU_NOT_SHARED] : Device->Host");

    printf("Step1: Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);

    int MAT_SIZE_A = ROW_SIZE * K_SIZE;
    int MAT_SIZE_B = K_SIZE * COL_SIZE;
    int MAT_SIZE_C = ROW_SIZE * COL_SIZE;

    // host input matrix
    float A[ROW_SIZE][K_SIZE]; // m * k
    float B[K_SIZE][COL_SIZE]; // k * n

    // host output matrix
    float hostC[ROW_SIZE][COL_SIZE];       // host result
    float deviceC[ROW_SIZE][COL_SIZE];     // device result
    float nsh_deviceC[ROW_SIZE][COL_SIZE]; // device result

    // device I/O matrix
    float *dA, *dB, *dC;
    dA = dB = dC = NULL;

    float *nsh_dA, *nsh_dB, *nsh_dC;
    nsh_dA = nsh_dB = nsh_dC = NULL;

    memset(A, 0, sizeof(float) * MAT_SIZE_A);
    memset(B, 0, sizeof(float) * MAT_SIZE_B);
    memset(hostC, 0, sizeof(float) * MAT_SIZE_C);
    memset(deviceC, 0, sizeof(float) * MAT_SIZE_C);

    memset(nsh_deviceC, 0, sizeof(float) * MAT_SIZE_C);

    // device memory allocaiton
    cudaMalloc(&dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&dC, sizeof(float) * MAT_SIZE_C);

    cudaMalloc(&nsh_dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&nsh_dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&nsh_dC, sizeof(float) * MAT_SIZE_C);

    // generate input matrices
    for (int r = 0; r < ROW_SIZE; r++)
        for (int k = 0; k < K_SIZE; k++)
            A[r][k] = ((rand() % 10) + ((rand() % 100) / 100.0));

    for (int k = 0; k < K_SIZE; k++)
        for (int c = 0; c < COL_SIZE; c++)
            B[k][c] = ((rand() % 10) + ((rand() % 100) / 100.0));

    // Host code
    printf("Step2: CPU Matrix Multiplication\n");
    timer.onTimer(CPU);
    for (int r = 0; r < ROW_SIZE; r++)
        for (int c = 0; c < COL_SIZE; c++)
            for (int k = 0; k < K_SIZE; k++)
                for (int i = 0; i < WORK_LOAD; i++)
                    hostC[r][c] += A[r][k] * B[k][c];
    timer.offTimer(CPU);

    // Copy input matrices : H -> D
    printf("Step3: CPU -> GPU \n");
    timer.onTimer(CPU2GPU);
    cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer.offTimer(CPU2GPU);

    //// Kernel call (shared memory)
    printf("Step4: GPU Matrix Mulatiplication\n");
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(COL_SIZE, ROW_SIZE);
    timer.onTimer(GPU);
    matMul_kernel_shared<<<gridDim, blockDim>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    timer.offTimer(GPU);

    // Get back result : D -> H
    printf("Step5: GPU -> CPU \n");
    timer.onTimer(GPU2CPU);
    cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer.offTimer(GPU2CPU);

    // check the results
    bool isCorrect = true;

    float *pHostC = &hostC[0][0];
    float *pDeviceC = &deviceC[0][0];

    for (int i = 0; i < MAT_SIZE_C; i++)
    {
        if (pHostC[i] != pDeviceC[i])
        {
            printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
            isCorrect = false;
            break;
        }
    }

    if (isCorrect)
        printf("SHARED Result is correct!\n");
    else
        printf("SHARED Result is not correct!!!!!!\n");

    timer.onTimer(NSH_CPU2GPU);
    cudaMemcpy(nsh_dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(nsh_dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer.offTimer(NSH_CPU2GPU);

    //// Kernel call (shared memory)
    timer.onTimer(NSH_GPU);
    matMul_kernel<<<gridDim, blockDim>>>(nsh_dA, nsh_dB, nsh_dC);
    cudaDeviceSynchronize();
    timer.offTimer(NSH_GPU);

    timer.onTimer(NSH_GPU2CPU);
    cudaMemcpy(nsh_deviceC, nsh_dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer.offTimer(NSH_GPU2CPU);

    float *nsh_pHostC = &hostC[0][0];
    float *nsh_pDeviceC = &nsh_deviceC[0][0];

    for (int i = 0; i < MAT_SIZE_C; i++)
    {
        if (nsh_pHostC[i] != nsh_pDeviceC[i])
        {
            printf("[%d] %.2f, %.2f\n", i, nsh_pHostC[i], nsh_pDeviceC[i]);
            isCorrect = false;
        }
    }

    if (isCorrect)
        printf("NOT SHARED Result is correct!\n");
    else
        printf("NOT SHARED Result is not correct!!!!!!\n");

    timer.printTimer();

    return 0;
}
