
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 8
//rai -p ece408_project --queue rai_amd64_ece408 --submit=m3

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int n = blockIdx.x;
    int m = blockIdx.y;
    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);
    int h = blockIdx.z/(H_Grid)*TILE_WIDTH + threadIdx.y; 
    int w = blockIdx.z%(W_Grid)*TILE_WIDTH + threadIdx.x;
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int c,p,q;
    if(h<H_out && w<W_out){
        // y4d(n, m, h, w) = 0;
        float a=0;
        //#pragma unroll
        for (c = 0; c < C; c++){
            #pragma unroll
            for (p = 0; p < K; p++)
                #pragma unroll
                for (q = 0; q < K; q++)
                    a+=x4d(n,c,h+p,w+q)*k4d(m, c, p, q);
        }
        y4d(n, m, h, w) = a;
    }

    
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
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
 
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int C = x.shape_[1];
    const int K = w.shape_[3];
    
    int H_out = H-K+1;
    int W_out = W-K+1;
    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);
    int Z=H_Grid*W_Grid;
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);
    dim3 gridDim(B,M,Z);
   
    // printf("B is: %d ", B);
    // printf("M is: %d ", M);
    // printf("H is: %d ", H);
    // printf("W is: %d ", W);
    // printf("C is: %d ", C);
    // printf("K is: %d ", K);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif