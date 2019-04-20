
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16


namespace mxnet {
namespace op {


__constant__ float kernel[6000];

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int n = blockIdx.x;
    int m = blockIdx.y;
    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);
    int h = blockIdx.z/(H_Grid)*TILE_WIDTH + threadIdx.y; 
    int w = blockIdx.z%(W_Grid)*TILE_WIDTH + threadIdx.x;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int c,p,q;
    if(h < H_out && w < W_out){
        y4d(n, m, h, w) = 0;
        for (c = 0; c < C; c++){
            #pragma unroll
            for (p = 0; p < K; p++){
                #pragma unroll
                for (q = 0; q < K; q++)
                    y4d(n,m,h,w) = y4d(n,m,h,w) + x4d(n,c,h+p,w+q) * k4d(m, c, p, q);
            }
        }
    }

    __syncthreads();

    #undef y4d
    #undef x4d
    #undef k4d
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

    int temp = w.shape_[2] * w.shape_[3] * w.shape_[0] * w.shape_[1];
    
    int H_out = H-K+1;
    int W_out = W-K+1;
    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);
    int Z= H_Grid * W_Grid;
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);
    dim3 gridDim(B,M,Z);
   
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