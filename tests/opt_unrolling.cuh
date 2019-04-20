
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
//rai -p ece408_project --queue rai_amd64_ece408 

namespace mxnet
{
namespace op
{

__global__ void opt_u(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  
  int bx=blockIdx.x; int by=blockIdx.y;
  int tx=threadIdx.x; int ty=threadIdx.y;
  int row,col;
  row=by*TILE_WIDTH+ty;
  col=bx*TILE_WIDTH+tx;
  float Pvalue=0;
  int count=0;
  for(int m=0; m<numAColumns/TILE_WIDTH;m++){
    subTileM[ty][tx]=A[row*numAColumns+m*TILE_WIDTH+tx];
    subTileN[ty][tx]=B[(m*TILE_WIDTH+ty)*numBColumns+col];
    __syncthreads();
    for(int k=0; k<TILE_WIDTH;k++){
      Pvalue+=subTileM[ty][k]*subTileN[k][tx];
      count=count+2;
    }
    __syncthreads();
  }
  int idx=row*numCColumns+col;
  printf("row: %d ", row);
  printf("col: %d", col);
  C[row*numCColumns+col]=Pvalue;
  
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
   

    // Call the kernel
    opt_u<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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