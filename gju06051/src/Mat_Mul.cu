#include "Mat_Mul.cuh"

__global__ void MatMul(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (_m <= row || _n <= col)
    {
        return; // finished job
    }

    float val = 0; // register
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    _matC[ID2INDEX(row, col, _n)] = val;
    return;
}

bool kernelCall(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k, dim3 _gridDim, dim3 _blockDim)
{
    MatMul<<<_gridDim, _blockDim>>>(_matA, _matB, _matC, _m, _n, _k);
    return true;
}