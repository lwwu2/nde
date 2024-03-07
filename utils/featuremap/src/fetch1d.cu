#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <stdexcept>

#include "include/cuda_math.h"
#include "include/cuda_helper.cuh"


/**
    Range query a 1D feature map given position and mip level
*/

// linear interpolate the feature of a single mip level
template <int C>
__device__ void lerp_1d(float* img, 
                        float xx, float r, 
                        int H, int T, 
                        float* f) {
    float x = xx*(H-1);

    if (x<0||x>(H-1)) {
        return;
    }

    int x0 = (int)floor(x);
    int x1 = (int)ceil(x);
    x-=x0;

    int idx[2] = {
        x0,x1
    };
    float w[2] = {
        (1-x),x
    };

    #pragma unroll
    for (int k=0; k<2; k++) {
        #pragma unroll
        for (int c=0; c<C; c++) {
            f[c] += w[k]*r*img[idx[k]+T*c];
        }
    }
}


// interpolate the 1D feature with mip support
template <int C>
__global__ void fetch_mip_1d_kernel(
    int32_t B,
    float* img, // CxN
    float* x_, // Bx1
    float* r_, // B
    int* offsets, // B
    const int resolution,
    const int T,
    float* ret // CxB
) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b>=B) {
        return;
    }

    x_ += b;
    r_ += b;
    ret += b;
    float x = x_[0];
    float r = r_[0];
    int r0 = int(floor(r));
    int r1 = int(ceil(r));
    r = r-r0;

    float f[C];
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] = 0.0f;
    }

    int offset = offsets[r0];
    int H = (int)(resolution/exp2f(r0));

    lerp_1d<C>(img+offset, x, 1-r, H, T, f);
    if (r!=0.0f) {
        offset = offsets[r1];
        H = (int)(resolution/exp2f(r1));
        lerp_1d<C>(img+offset, x, r, H, T, f);
    }

    #pragma unroll
    for (int c=0; c<C; c++) {
        ret[B*c] = f[c];
    }
}



// gradient of linear interpolation respect to feature and position
template <int C>
__device__ float lerp_1d_bwd(float* img, float* g,
            float xx, float r,
            int H, int T,
            float* dx, float* img_grad) {
    float x = xx*(H-1);
    if (x<0||x>(H-1)) {
        return 0.0f;
    }
    int x0 = (int)floor(x);
    int x1 = (int)ceil(x);

    x -= x0;

    int idx[2] = {x0,x1};
    float gf[2] = {0.0f,0.0f,};
    float w[2] = {(1-x),x};

    #pragma unroll
    for (int k=0; k<2; k++) {
        #pragma unroll
        for (int c=0; c<C; c++) {
            gf[k] += g[c]*img[idx[k]+c*T];
        }
    }

    if (r!=0.0f) {
        #pragma unroll
        for (int k=0; k<2; k++) {
            #pragma unroll
            for (int c=0; c<C; c++) {
                atomicAdd(img_grad+idx[k]+c*T,w[k]*r*g[c]);
            }
        }
    }

    float dr = 0.0f;
    #pragma unroll
    for (int k=0; k<2; k++) {
        dr += gf[k]*w[k];
    }

    if (r!=0.0f) {
        //dx
        w[0] = -1.0f;w[1] = 1.0f;
        #pragma unroll
        for (int k=0; k<2; k++) {
            dx[0] += w[k]*gf[k]*r*(H-1);
        }
    }
    return dr;
}


// gradient of feature query respect to feature, position, and mip level
template <int C>
__global__ void fetch_mip_1d_bwd_kernel(
    int32_t B,
    float* grad_in, //CxB
    float* img, // CxN
    float* x_, // Bx2
    float* r_, // B
    int* offsets, // B
    const int resolution,
    const int T,
    float* img_grad,
    float* x_grad,
    float* r_grad
) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b>=B) {
        return;
    }

    grad_in += b;
    x_ += b;
    r_ += b;
    x_grad += b;
    r_grad += b;

    float g[C];
    #pragma unroll
    for (int c=0; c<C; c++) {
        g[c] = grad_in[c*B];
    }

    float x = x_[0];
    float r = r_[0];
    
    int r0 = int(floor(r));
    int r1 = int(ceil(r));
    r = r-r0;

    float dx = 0.0f;

    int offset = offsets[r0];
    int H = (int)(resolution/exp2f(r0));

    float dr = -lerp_1d_bwd<C>(img+offset, g,x, 1-r, H, T, &dx, img_grad+offset);
    
    offset = offsets[r1];
    H = (int)(resolution/exp2f(r1));
    dr += lerp_1d_bwd<C>(img+offset,g,x,r,H,T,&dx,img_grad+offset);

    x_grad[0] = dx;
    r_grad[0] = dr;
}





/////////////////////////////////////////////////////////////////////
// only support feature size of 3, 4, 8, 16, 32

void fetch_mip_1d(
    torch::Tensor img,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor ret
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t C = img.size(0);
    int32_t B = x.size(0);
    int32_t T = img.size(1);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_1d_kernel<3><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;

        case 4: fetch_mip_1d_kernel<4><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        
        case 8: fetch_mip_1d_kernel<8><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        
        case 16: fetch_mip_1d_kernel<16><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        case 32: fetch_mip_1d_kernel<32><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature 1D: C must be 3, 4, 8, 16, 32"};
    }
}


void fetch_mip_1d_bwd(
    torch::Tensor grad_in,
    torch::Tensor img,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor img_grad,
    torch::Tensor x_grad,
    torch::Tensor r_grad
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t C = img.size(0);
    int32_t B = x.size(0);
    int32_t T = img.size(1);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_1d_bwd_kernel<3><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;

        case 4: fetch_mip_1d_bwd_kernel<4><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 8: fetch_mip_1d_bwd_kernel<8><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 16: fetch_mip_1d_bwd_kernel<16><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        case 32: fetch_mip_1d_bwd_kernel<32><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            x.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            x_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature 1D: C must be 3, 4, 8, 16, 32"};
    }
}