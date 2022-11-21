#include "Mat_Mul.cuh"
#include "DS_timer.cuh"

#define SIZE_M (512 * 2)
#define SIZE_N (512 * 4)
#define SIZE_K (512 * 2)

bool MatMul_GPU(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k, dim3 _gridDim, dim3 _blockDim);

int main(int argc, char *argv[])
{
    // timer set
    DS_timer timer(4);
    timer.setTimerName(0, (char *)"[CPU]");
    timer.setTimerName(1, (char *)"[GPU]");
    timer.setTimerName(2, (char *)"[DATA Transfer] : Host->Device");
    timer.setTimerName(3, (char *)"[DATA Transfer] : Device->Host");

    // get matrix size spec
    // invalid argument, use default (1024_1024) x (1024_2048)
    int m, n, k;
    if (argc < 3) // default argument
    {
        m = SIZE_M;
        n = SIZE_N;
        k = SIZE_K;
    }
    else // argument user give
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    printf("Step1: Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", m, k, k, n, m, n);

    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    // CPU matrix generation
    float *h_A, *h_B, *h_C, *gpu_C;

    h_A = new float[sizeA];
    h_B = new float[sizeB];
    h_C = new float[sizeC];
    gpu_C = new float[sizeC];

    memset(h_A, 0, sizeA);
    memset(h_B, 0, sizeB);
    memset(h_C, 0, sizeC);
    memset(gpu_C, 0, sizeC);

    for (int i = 0; i < sizeA; i++)
    {
        h_A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    for (int i = 0; i < sizeB; i++)
    {
        h_B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    printf("Step2: CPU Matrix generation finished\n");

    // CPU MatMul
    timer.onTimer(0);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            int c_idx = ID2INDEX(row, col, n);
            h_C[c_idx] = 0;
            for (int i = 0; i < k; i++)
            {
                h_C[c_idx] += (h_A[ID2INDEX(row, i, k)] * h_B[ID2INDEX(i, col, n)]);
            }
        }
    }
    timer.offTimer(0);
    printf("Step3: CPU MatMul finished\n");

    // GPU matrix generation
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemset(d_A, 0, sizeA * sizeof(float));
    cudaMemset(d_B, 0, sizeB * sizeof(float));
    cudaMemset(d_C, 0, sizeC * sizeof(float));

    timer.onTimer(2);
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    timer.offTimer(2);

    printf("Step4: GPU matrix generation finished\n");

    // grid, block setting
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Step6: Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // GPU Matmul
    timer.onTimer(1);
    MatMul_GPU(d_A, d_B, d_C, m, n, k, gridDim, blockDim);
    cudaDeviceSynchronize();
    timer.offTimer(1);
    printf("Step7: GPU matrix multiplication finished\n");

    timer.onTimer(3);
    cudaMemcpy(gpu_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    printf("Step8: GPU result transfer to CPU finished\n");

    bool result = true;
    for (int i = 0; i < sizeC; i++)
    {
        if (h_C[i] != gpu_C[i])
        {
            printf("[%d] not matched! (%f, %f)\n", i, h_C[i], gpu_C[i]);
            result = false;
        }
    }
    if (result)
    {
        printf("GPU work well!\n");
    }
    timer.printTimer();

    return result;
}

bool MatMul_GPU(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k, dim3 _gridDim, dim3 _blockDim)
{
    return kernelCall(_matA, _matB, _matC, _m, _n, _k, _gridDim, _blockDim);
}