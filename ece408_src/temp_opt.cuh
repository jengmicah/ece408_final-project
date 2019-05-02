
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define TILE_WIDTH2 24
// #define TILE_WIDTH_2 24
// #define TILE_WIDTH_3 15

//16,8
//16,24
//

/* 

    ./rai -p ece408_project --queue rai_amd64_ece408

*/

namespace mxnet {
namespace op {


__constant__ float kernel[6000];

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    //load W into the shared memory 
    // all threads collaborate to copy portion of the input X that is required to compute the output tile into the shared memory 
    // compute partial sum of output_y 
    // move to the next input channel 

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int x_tile_width  = TILE_WIDTH + K -1;

    int n = blockIdx.x; //which input image 
    int m = blockIdx.y; //which mask 

    int H_Grid = ceil(H_out/(float)TILE_WIDTH);
    int W_Grid = ceil(W_out/(float)TILE_WIDTH);

    int h0 = threadIdx.y; 
    int w0 = threadIdx.x;

    int h_base = (blockIdx.z / H_Grid)*TILE_WIDTH; //vertical base out data index for this block 
    int w_base = (blockIdx.z % W_Grid)*TILE_WIDTH; //horizontal base out data index for the block

    int h = h_base + h0; 
    int w = w_base+ w0;
    // const int feature_width = K;
    extern __shared__ float shmem[];

    float* shared_feature_map = &shmem[x_tile_width* x_tile_width];
    float* shared_x = &shmem[0];

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0;
    for(int c = 0; c < C; c++){
        // for each channel, 
        if((h0 < K ) && (w0 <K)){
            // copy corresponding element into the shared mem from feature map based on which channel 
            shared_feature_map[h0 * K + w0] = k4d(m,c,h0,w0);
        }
        __syncthreads();
        // #pragma unroll 2
        for(int i = h; i < h_base + x_tile_width; i += TILE_WIDTH){
            // copy the portion of input into shared meme 
            if(i < H){
                #pragma unroll 
                for(int j = w; j < w_base +x_tile_width && j < W; j+= TILE_WIDTH){
                    if(j < W){
                        shared_x[(i-h_base)*x_tile_width + (j-w_base)] = x4d(n,c,i,j); 
                    }
                }
            }
        }
        
        __syncthreads();
        
        for(int p = 0; p < K; p++){
            #pragma unroll 
            for(int q = 0; q < K; q++){
                //shared_x[h0+p, w0+q]
                acc += shared_x[(h0+p)*x_tile_width + (w0+q)]* shared_feature_map[p*K + q];
            }
        }
        __syncthreads();
        
        // __syncthreads();
        // compute the partial sum of the output 
    }
    if(h < H_out && w < W_out){
        y4d(n,m,h,w) = acc;
    }
    
    #undef y4d
    #undef x4d
    #undef k4d
}

__global__ void forward_kernel2(float *__restrict__ y, float *__restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = blockIdx.z;
    int width = C * K * K;
    int col = tx + blockIdx.x * TILE_WIDTH2;
    int row = ty + blockIdx.y * TILE_WIDTH2; 

    // input matrix
    int matXRows = width;
    int matXCols = H_out * W_out;

    // weight matrix 
    int matKRows = M;
    int matKCols = width;

    // output matrix
    int matYRows = M;
    int matYCols = H_out * W_out;

    __shared__ float matX[TILE_WIDTH2][TILE_WIDTH2];
    __shared__ float matK[TILE_WIDTH2][TILE_WIDTH2];

    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    float acc = 0;
    int xh, xw, xc, xq, xp, xt;

    // load kernel into shared
    #pragma unroll 2
    for(int j = 0; j < (width + TILE_WIDTH2 - 1) / TILE_WIDTH2; j ++) {
       
        if(j * TILE_WIDTH2 + tx < width && row < matYRows) {
            matK[ty][tx] = kernel[row * matKCols + j * TILE_WIDTH2 + tx]; 
        } else {
            matK[ty][tx] = 0;
        }

        if(j * TILE_WIDTH2 + ty < width && col < matYCols) {
            xh = col / W_out;
            xw = col % W_out;
            xc = (j * TILE_WIDTH2 + ty) / (K * K);
            xt = (j * TILE_WIDTH2 + ty) % (K * K);
            xp = xt / K;
            xq = xt % K;
            matX[ty][tx] = x4d(bz, xc, xh + xp, xw + xq);
        } else {
            matX[ty][tx] = 0;
        }

        __syncthreads();

        #pragma unroll 16
        for(int i = 0; i < TILE_WIDTH2; i++) {
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
    
    // printf("B is: %d ", B);
    // printf("M is: %d ", M);
    // printf("H is: %d ", H);
    // printf("W is: %d ", W);
    // printf("C is: %d ", C);
    // printf("K is: %d ", K);

    int temp = w.shape_[0] * w.shape_[1] * w.shape_[2] * w.shape_[3];
    
    int H_out = H-K+1;
    int W_out = W-K+1;
    cudaMemcpyToSymbol(kernel, w.dptr_, temp * sizeof(float), 0, cudaMemcpyDefault);
    if(W%16==0){
        int H_Grid = ceil(H_out/(float)TILE_WIDTH);
        int W_Grid = ceil(W_out/(float)TILE_WIDTH);
        int Z=H_Grid*W_Grid;
        size_t shmem_size = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1) + K*K);
        dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);
        dim3 gridDim(B,M,Z);
        
        forward_kernel<<<gridDim, blockDim,shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    }
   else{
        int H_Grid = ceil(H_out/(float)TILE_WIDTH2);
        int W_Grid = ceil(W_out/(float)TILE_WIDTH2);
        // int Z= H_Grid * W_Grid;

        dim3 blockDim(TILE_WIDTH2,TILE_WIDTH2, 1);
        dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH2), ceil(1.0*M/TILE_WIDTH2), B);
        // cudaMemcpyToSymbol(kernel2, w.dptr_, temp * sizeof(float), 0, cudaMemcpyDefault);

        // Call the kernel
        
        forward_kernel2<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
    }
        

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