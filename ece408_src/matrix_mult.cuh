
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define TILE_WIDTH_2 24
#define TILE_WIDTH_3 15


/* 

    ./rai -p ece408_project --queue rai_amd64_ece408

*/

namespace mxnet {
namespace op {


__constant__ float kernel[6000];

__global__ void forward_kernel(float *y, float *x, const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = blockIdx.z;
    int width = C * K * K;

    int col = tx + blockIdx.x * TILE_WIDTH;
    int row = ty + blockIdx.y * TILE_WIDTH; 

    // input matrix
    int matXRows = width;
    int matXCols = H_out * W_out;

    // weight matrix 
    int matKRows = M;
    int matKCols = width;

    // output matrix
    int matYRows = M;
    int matYCols = H_out * W_out;

    __shared__ float matX[TILE_WIDTH][TILE_WIDTH];
    __shared__ float matK[TILE_WIDTH][TILE_WIDTH];

    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    float acc = 0;
    int xh, xw, xc, xq, xp, xt;

    // load kernel into shared
    #pragma unroll
    for(int j = 0; j < (width + TILE_WIDTH - 1) / TILE_WIDTH; j ++) {
       
        if(j * TILE_WIDTH + tx < width && row < matYRows) {
            matK[ty][tx] = kernel[row * matKCols + j * TILE_WIDTH + tx]; 
        } else {
            matK[ty][tx] = 0;
        }

        if(j * TILE_WIDTH + ty < width && col < matYCols) {
            xh = col / W_out;
            xw = col % W_out;
            xc = (j * TILE_WIDTH + ty) / (K * K);
            xt = (j * TILE_WIDTH + ty) % (K * K);
            xp = xt / K;
            xq = xt % K;
            matX[ty][tx] = x4d(bz, xc, xh + xp, xw + xq);
        } else {
            matX[ty][tx] = 0;
        }

        __syncthreads();

        #pragma unroll
        for(int i = 0; i < TILE_WIDTH; i++) {
            acc += (matK[ty][i] * matX[i][tx]);
        }

        __syncthreads();
    }


    if(row < matYRows && col < matYCols) {
        y[(bz * M * H_out * W_out) + row*matYCols + col] = acc;
    }

    #undef x4d
}



/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {

    /*
        x   Input data  batch size * input channels * y * x
        y   Output data     batch size * output channels * y * x
        k   kernel weights  output channels * input channels * y * x
    */

    const int B = x.shape_[0]; //batch size
    const int M = y.shape_[1]; //output channels
    const int H = x.shape_[2]; //y
    const int W = x.shape_[3]; //x
    const int C = x.shape_[1]; //input channels
    const int K = w.shape_[3]; //filter x

    int temp = w.shape_[0] * w.shape_[1] * w.shape_[2] * w.shape_[3];
    
    int H_out = H-K+1;
    int W_out = W-K+1;
    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);
    int Z= H_Grid * W_Grid;


    dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);
    // dim3 gridDim(B,M,Z);
    dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH), B);


    cudaMemcpyToSymbol(kernel, w.dptr_, temp * sizeof(float), 0, cudaMemcpyDefault);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif