#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <stdexcept>

#include "include/cuda_math.h"
#include "include/cuda_helper.cuh"


/**
    Range query a 2D feature map given position and mip level
*/

// bilinear interpolation of single mip level
template <int C>
__device__ void lerp_2d(float* img, 
                        float2 xy, float r, 
                        int H, int T, 
                        float* f) {
    float x = xy.x*(H-1);
    float y = xy.y*(H-1);

    if (x<0||y<0||x>(H-1)||y>(H-1)) {
        return;
    }

    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = (int)ceil(x);
    int y1 = (int)ceil(y);
    x-=x0;
    y-=y0;
    float _x = 1-x;
    float _y = 1-y;

    int idx[4] = {
        x0+y0*H,x1+y0*H,
        x0+y1*H,x1+y1*H
    };
    float w[4] = {
        _x*_y,x*_y,
        _x*y,x*y
    };

    #pragma unroll
    for (int k=0; k<4; k++) {
        #pragma unroll
        for (int c=0; c<C; c++) {
            f[c] += w[k]*r*img[idx[k]+T*c];
        }
    }
}


// trilinear interpolate a 2d feature map query
template <int C>
__global__ void fetch_mip_2d_kernel(
    int32_t B,
    float* img, // CxN
    float* xy_, // Bx2
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

    xy_ += b*2;
    r_ += b;
    ret += b;
    float2 xy = make_float2(xy_[0],xy_[1]);
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

    lerp_2d<C>(img+offset, xy, 1-r, H, T, f);
    if (r!=0.0f) {
        offset = offsets[r1];
        H = (int)(resolution/exp2f(r1));
        lerp_2d<C>(img+offset, xy, r, H, T, f);
    }

    #pragma unroll
    for (int c=0; c<C; c++) {
        ret[B*c] = f[c];
    }
}

// gradiet of bilinear interpolation respect to position and feature
template <int C>
__device__ float lerp_2d_bwd(float* img, float* g,
            float2 xy, float r,
            int H, int T,
            float* dxy, float* img_grad) {
    float x = xy.x*(H-1);
    float y = xy.y*(H-1);
    if (x<0||y<0||x>(H-1)||y>(H-1)) {
        return 0.0f;
    }
    int x0 = (int)floor(x);
    int x1 = (int)ceil(x);
    int y0 = (int)floor(y);
    int y1 = (int)ceil(y);

    x -= x0;
    y -= y0;

    float _x = (1-x);
    float _y = (1-y);

    int idx[4] = {x0+y0*H,x1+y0*H,x0+y1*H,x1+y1*H};
    float gf[4] = {0.0f,0.0f,0.0f,0.0f};
    float w[4] = {_x*_y,x*_y,_x*y,x*y};

    #pragma unroll
    for (int k=0; k<4; k++) {
        #pragma unroll
        for (int c=0; c<C; c++) {
            gf[k] += g[c]*img[idx[k]+c*T];
        }
    }

    if (r!=0.0f) {
        #pragma unroll
        for (int k=0; k<4; k++) {
            #pragma unroll
            for (int c=0; c<C; c++) {
                atomicAdd(img_grad+idx[k]+c*T,w[k]*r*g[c]);
            }
        }
    }

    float dr = 0.0f;
    #pragma unroll
    for (int k=0; k<4; k++) {
        dr += gf[k]*w[k];
    }

    if (r!=0.0f) {
        //dx
        w[0] = -_y;w[1] = _y;
        w[2] = -y;w[3] = y;
        #pragma unroll
        for (int k=0; k<4; k++) {
            dxy[0] += w[k]*gf[k]*r*(H-1);
        }

        //dy
        w[0] = -_x;w[1]=-x;
        w[2] = _x;w[3] = x;
        #pragma unroll
        for (int k=0; k<4; k++) {
            dxy[1] += w[k]*gf[k]*r*(H-1);
        }
    }
    return dr;
}

// gradient of feature map query respect feature, position, mip level
template <int C>
__global__ void fetch_mip_2d_bwd_kernel(
    int32_t B,
    float* grad_in, //CxB
    float* img, // CxN
    float* xy_, // Bx2
    float* r_, // B
    int* offsets, // B
    const int resolution,
    const int T,
    float* img_grad,
    float* xy_grad,
    float* r_grad
) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b>=B) {
        return;
    }

    grad_in += b;
    xy_ += b*2;
    r_ += b;
    xy_grad += b*2;
    r_grad += b;

    float g[C];
    #pragma unroll
    for (int c=0; c<C; c++) {
        g[c] = grad_in[c*B];
    }

    float2 xy = make_float2(xy_[0],xy_[1]);
    float r = r_[0];
    
    int r0 = int(floor(r));
    int r1 = int(ceil(r));
    r = r-r0;

    float dxy[2] = {0.0f,0.0f};

    int offset = offsets[r0];
    int H = (int)(resolution/exp2f(r0));

    float dr = -lerp_2d_bwd<C>(img+offset, g,xy, 1-r, H, T, dxy, img_grad+offset);
    
    offset = offsets[r1];
    H = (int)(resolution/exp2f(r1));
    dr += lerp_2d_bwd<C>(img+offset,g,xy,r,H,T,dxy,img_grad+offset);

    xy_grad[0] = dxy[0];
    xy_grad[1] = dxy[1];
    r_grad[0] = dr;
}







////////////////////////////////////////////////////////////////////////////
// only support feature size of 3, 4, 8, 16, 32


void fetch_mip_2d(
    torch::Tensor img,
    torch::Tensor xy,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor ret
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t C = img.size(0);
    int32_t B = xy.size(0);
    int32_t T = img.size(1);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_2d_kernel<3><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;

        case 4: fetch_mip_2d_kernel<4><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        
        case 8: fetch_mip_2d_kernel<8><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        
        case 16: fetch_mip_2d_kernel<16><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        case 32: fetch_mip_2d_kernel<32><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            ret.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature 2D: C must be 3, 4, 8, 16, 32"};
    }
}


void fetch_mip_2d_bwd(
    torch::Tensor grad_in,
    torch::Tensor img,
    torch::Tensor xy,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor img_grad,
    torch::Tensor xy_grad,
    torch::Tensor r_grad
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t C = img.size(0);
    int32_t B = xy.size(0);
    int32_t T = img.size(1);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_2d_bwd_kernel<3><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            xy_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;

        case 4: fetch_mip_2d_bwd_kernel<4><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            xy_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 8: fetch_mip_2d_bwd_kernel<8><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            xy_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 16: fetch_mip_2d_bwd_kernel<16><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            xy_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        case 32: fetch_mip_2d_bwd_kernel<32><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            xy.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,T,
            img_grad.data_ptr<float>(),
            xy_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature 2D: C must be 3, 4, 8, 16, 32"};
    }
}