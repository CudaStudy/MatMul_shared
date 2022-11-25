#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAT_MUL_NSH
// #define MAT_MUL_SH

// Matrix Spec
#define ROW_SIZE 1024
#define K_SIZE 1024
#define COL_SIZE 2048
#define WORK_LOAD (ROW_SIZE * COL_SIZE)

#define MAT_SIZE_A (ROW_SIZE * K_SIZE)
#define MAT_SIZE_B (K_SIZE * COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE * COL_SIZE)

#define BLOCK_SIZE 16

// Macro
#define INDEX2ROW(_index, _width) (int)((_index) / (_width))
#define INDEX2COL(_index, _width) ((_index) % (_width))
#define ID2INDEX(_row, _col, _width) (((_row) * (_width)) + (_col))

#define CPU 0

#ifdef MAT_MUL_SH
#define GPU 0
#define CPU2GPU 1
#define GPU2CPU 2
#endif

#ifdef MAT_MUL_NSH
#define NSH_GPU 0
#define NSH_CPU2GPU 1
#define NSH_GPU2CPU 2
#endif

#ifdef MAT_MUL_SH
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
#endif

#ifdef MAT_MUL_NSH
__global__ void matMul_kernel(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= _m || col >= _n)
        return;

    float val = 0;
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    _matC[ID2INDEX(row, col, _n)] = val;
}
#endif

int main(void)
{
    DS_timer timer_cpu(1);
    timer_cpu.setTimerName(CPU, (char *)"[CPU]");

#ifdef MAT_MUL_SH
    DS_timer timer_sh(3);
    timer_sh.setTimerName(GPU, (char *)"[GPU_SHARED] : Multilplication");
    timer_sh.setTimerName(CPU2GPU, (char *)"[GPU_SHARED] : Host->Device");
    timer_sh.setTimerName(GPU2CPU, (char *)"[GPU_SHARED] : Device->Host");
#endif

#ifdef MAT_MUL_NSH
    DS_timer timer_nsh(3);
    timer_nsh.setTimerName(NSH_GPU, (char *)"[GPU_NOT_SHARED] : Multiplication");
    timer_nsh.setTimerName(NSH_CPU2GPU, (char *)"[GPU_NOT_SHARED] : Host->Device");
    timer_nsh.setTimerName(NSH_GPU2CPU, (char *)"[GPU_NOT_SHARED] : Device->Host");
#endif

    printf("Step1: Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);

#ifdef MAT_MUL_SH
    // host input matrix
    float *A = new float[ROW_SIZE][K_SIZE];       // m * k
    float *B = new float[K_SIZE][COL_SIZE];       // k * n
    float *hostC = new float[ROW_SIZE][COL_SIZE]; // host result

    memset(A, 0, sizeof(float) * MAT_SIZE_A);
    memset(B, 0, sizeof(float) * MAT_SIZE_B);
    memset(hostC, 0, sizeof(float) * MAT_SIZE_C);

    // generate input matrices
    for (int r = 0; r < ROW_SIZE; r++)
        for (int k = 0; k < K_SIZE; k++)
            A[r][k] = ((rand() % 10) + ((rand() % 100) / 100.0));

    for (int k = 0; k < K_SIZE; k++)
        for (int c = 0; c < COL_SIZE; c++)
            B[k][c] = ((rand() % 10) + ((rand() % 100) / 100.0));

    // Host code
    printf("Step2: CPU Matrix Multiplication\n");
    timer_sh.onTimer(CPU);
    for (int r = 0; r < ROW_SIZE; r++)
        for (int c = 0; c < COL_SIZE; c++)
            for (int k = 0; k < K_SIZE; k++)
                for (int i = 0; i < WORK_LOAD; i++)
                    hostC[r][c] += A[r][k] * B[k][c];
    timer_sh.offTimer(CPU);

    // device result
    float deviceC[ROW_SIZE][COL_SIZE];
    memset(deviceC, 0, sizeof(float) * MAT_SIZE_C);

    // device I/O matrix
    float *dA, *dB, *dC;
    dA = dB = dC = NULL;

    // device memory allocaiton
    cudaMalloc(&dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&dC, sizeof(float) * MAT_SIZE_C);

#endif

#ifdef MAT_MUL_NSH
    // host input matrix
    float *A = new float[MAT_SIZE_A];
    float *B = new float[MAT_SIZE_B];
    float *hostC = new float[MAT_SIZE_C]; // host result

    memset(A, 0, sizeof(float) * MAT_SIZE_A);
    memset(B, 0, sizeof(float) * MAT_SIZE_B);
    memset(hostC, 0, sizeof(float) * MAT_SIZE_C);

    // generate input matrices
    for (int i = 0; i < MAT_SIZE_A; i++)
    {
        A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    for (int i = 0; i < MAT_SIZE_B; i++)
    {
        B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }

    printf("Step2: CPU Matrix Multiplication\n");
    timer_cpu.onTimer(CPU);
    for (int row = 0; row < ROW_SIZE; row++)
    {
        for (int col = 0; col < COL_SIZE; col++)
        {
            int matC_idx = ID2INDEX(row, col, COL_SIZE);
            hostC[matC_idx] = 0;
            for (int i = 0; i < K_SIZE; i++)
            {
                hostC[matC_idx] += A[ID2INDEX(row, i, K_SIZE)] * B[ID2INDEX(i, col, COL_SIZE)];
            }
        }
    }
    timer_cpu.offTimer(CPU);

    // device result
    float *nsh_deviceC = new float[MAT_SIZE_C];
    memset(nsh_deviceC, 0, sizeof(float) * MAT_SIZE_C);

    // device I/O matrix
    float *nsh_dA, *nsh_dB, *nsh_dC;
    nsh_dA = nsh_dB = nsh_dC = NULL;

    // device memory allocation
    cudaMalloc(&nsh_dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&nsh_dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&nsh_dC, sizeof(float) * MAT_SIZE_C);

#endif

    printf("--------------------CPU MULTIPLICATION TOP--------------------\n");
    timer_cpu.printTimer();

    // check the results
    bool isCorrect = true;
    printf("--------------------CPU MULTIPLICATION BOT--------------------\n");

#ifdef MAT_MUL_SH
    // Copy input matrices : H -> D
    printf("Step3: CPU -> GPU \n");
    timer_sh.onTimer(CPU2GPU);
    cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer_sh.offTimer(CPU2GPU);

    //// Kernel call (shared memory)
    printf("Step4: GPU Matrix Mulatiplication\n");

    dim3 gridDim_sh(1);
    dim3 blockDim_sh(ROW_SIZE, COL_SIZE);
    timer_sh.onTimer(GPU);
    matMul_kernel_shared<<<gridDim_sh, blockDim_sh>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    timer_sh.offTimer(GPU);

    // Get back result : D -> H
    printf("Step5: GPU -> CPU \n");
    timer_sh.onTimer(GPU2CPU);
    cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer_sh.offTimer(GPU2CPU);

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

    printf("--------------------GPU_SH MULTIPLICATION TOP--------------------\n");
    timer_sh.printTimer();
    printf("--------------------GPU_SH MULTIPLICATION BOT--------------------\n");

#endif

#ifdef MAT_MUL_NSH
    timer_nsh.onTimer(NSH_CPU2GPU);
    cudaMemcpy(nsh_dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(nsh_dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer_nsh.offTimer(NSH_CPU2GPU);

    int m = ROW_SIZE;
    int n = COL_SIZE;
    int k = K_SIZE;

    dim3 gridDim_nsh(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim_nsh(BLOCK_SIZE, BLOCK_SIZE);
    printf("Step6: not_shared gpu multiplication start!\n");
    printf("Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);
    printf("Dim  : Grid = (%d, %d), BLock = (%d x %d)\n", gridDim_nsh.x, gridDim_nsh.y, blockDim_nsh.x, blockDim_nsh.y);

    //// Kernel call (not shared memory)
    timer_nsh.onTimer(NSH_GPU);
    matMul_kernel<<<gridDim_nsh, blockDim_nsh>>>(nsh_dA, nsh_dB, nsh_dC, m, n, k);
    cudaDeviceSynchronize();
    timer_nsh.offTimer(NSH_GPU);

    timer_nsh.onTimer(NSH_GPU2CPU);
    cudaMemcpy(nsh_deviceC, nsh_dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer_nsh.offTimer(NSH_GPU2CPU);

    float *nsh_pHostC = &hostC[0];
    float *nsh_pDeviceC = &nsh_deviceC[0];
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

    printf("--------------------GPU_NSH MULTIPLICATION TOP--------------------\n");
    timer_nsh.printTimer();
    printf("--------------------GPU_NSH MULTIPLICATION BOT--------------------\n");
#endif

    return 0;
}
