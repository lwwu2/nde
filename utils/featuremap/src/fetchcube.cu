#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <stdexcept>

#include "include/cuda_math.h"
#include "include/cuda_helper.cuh"

/**
    cubemap-direction coordinate transformation
    following convention:
    [(1,2),(2,0),(0,1)]
    (1,u,v),(u,1,v),(u,v,1)
    (-1,-u,-v),(-u,-1,-v),(-u,-v,-1)
*/

__device__ float3 cube2xyz(int j, int i, int side, int N) {
    float u = 2.0f * (((float)i + 0.5f) / (float)N) - 1.0f;
    float v = 2.0f * (((float)j + 0.5f) / (float)N) - 1.0f;

    switch (side) {
        case 0: return safeNormalize(make_float3(1.0f,u,v));
        case 1: return safeNormalize(make_float3(u,1.0f,v));
        case 2: return safeNormalize(make_float3(u,v,1.0f));
        case 3: return safeNormalize(make_float3(-1.0f,u,v));
        case 4: return safeNormalize(make_float3(u,-1.0f,v));
        case 5: return safeNormalize(make_float3(u,v,-1.0f));
    }
    return make_float3(0,0,0);
}

__device__ float3 xyz2cube(float3 wi) {
    float tmax = abs(wi.x);
    float3 uvw;
    if (tmax>abs(wi.y)) {
        if (tmax>abs(wi.z)) {
            uvw.z = wi.x>0? 0:3;
            uvw.x=wi.y/tmax;
            uvw.y=wi.z/tmax;
        } else {
            tmax = abs(wi.z);
            uvw.z = wi.z>0? 2:5;
            uvw.x = wi.x/tmax;
            uvw.y = wi.y/tmax;
        }
    } else {
        tmax = abs(wi.y);
        if (tmax>abs(wi.z)) {
            uvw.z = wi.y>0? 1:4;
            uvw.x = wi.x/tmax;
            uvw.y = wi.z/tmax;
        } else {
            tmax = abs(wi.z);
            uvw.z = wi.z>0? 2:5;
            uvw.x = wi.x/tmax;
            uvw.y = wi.y/tmax;
        }
    }
    uvw.x = uvw.x*0.5+0.5;
    uvw.y = uvw.y*0.5+0.5;
    return uvw;
}


////////////////////////////////////////////////////////////////////////////
/** 
    filtering cubemap by a GGX kernel,
    modified from https://github.com/NVlabs/nvdiffrec/blob/a3e73909a01887c8a135235ff860dd23a045cc1b/render/renderutils/c_src/cubemap.cu
*/
/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 */

// GGX kernel
__device__ inline float ndfGGX(const float alphaSqr, const float cosTheta) {
    float _cosTheta = clamp(cosTheta, 0.0, 1.0f);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * M_PI);
}

// area of a pixel
__device__ float pixel_area(int x, int y, int N) {
    if (N > 1)
    {
        int H = N / 2;
        x = abs(x - H);
        y = abs(y - H);
        float dx = atan((float)(x + 1) / (float)H) - atan((float)x / (float)H);
        float dy = atan((float)(y + 1) / (float)H) - atan((float)y / (float)H);
        return dx * dy;
    }
    else
        return 1;
}

// find filter bound for each cubemap
__global__ void specular_bound_kernel(
    const int resolution,
    const float cutoff,
    int* ret //(6x4)x6xHxW
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;

    if (px>=resolution || py>=resolution || pz>=6) {
        return;
    }

    float3 normal = cube2xyz(px,py,pz,resolution);
    const int TILE_SIZE = 16;
    ret += pz*resolution*resolution+py*resolution+px;

    for (int s=0; s < 6; s++) {
        int _min_x = resolution-1,_max_x = 0;
        int _min_y = resolution-1,_max_y = 0;
        
        for (int ii=0; ii < div_round_up(resolution,TILE_SIZE); ii++) {
            for (int jj=0; jj < div_round_up(resolution,TILE_SIZE); jj++) {
                for (int i_=0; i_<TILE_SIZE; i_++) {
                    int i = ii*TILE_SIZE+i_;
                    if (i>=resolution) {break;}
                    for (int j_=0; j_<TILE_SIZE; j_++) {
                        int j = jj*TILE_SIZE+j_;
                        if (j>=resolution) {break;}

                        float3 wi = cube2xyz(j,i,s,resolution);
                        float cos_theta = dot(wi,normal);
                        if (cos_theta>=cutoff) {
                            _min_x = min(_min_x,j);
                            _max_x = max(_max_x,j);
                            _min_y = min(_min_y,i);
                            _max_y = max(_max_y,i);
                        }
                    }
                }
            }
        }
        int offset = 6*resolution*resolution;
        ret[0] = _min_x;
        ret[offset] = _max_x;
        ret[offset*2] = _min_y;
        ret[offset*3] = _max_y;
        ret += 4*offset;
    }
}

// filteing the cubemap by a GGX kernel
__global__ void specular_filtering_kernel(
    const int H, const int C, const float roughness,
    const float* img, // Cx6xHxW
    const int* bound, // (6x4)x6xHxW
    float* ret, // Cx6xHxW
    float* weight //6xHxW
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;

    if (px>=H || py>=H || pz>=6) {
        return;
    }

    float alpha2 = roughness*roughness;
    alpha2 = alpha2*alpha2;
    float3 normal = cube2xyz(px,py,pz,H);
    int idx = px + py*H + pz*H*H;
    bound += idx;
    ret += idx;
    weight += idx;


    float col[16];
    #pragma unroll
    for (int c=0; c<C; c++) {
        col[c] = 0.0f;
    }
    float wsum = 0.0f;
    for (int s=0; s<6; s++) {
        int bound_[4];
        #pragma unroll 4
        for (int b=0;b<4;b++){
            bound_[b] = bound[b*6*H*H];
        }
        bound += 24*H*H;

        if (bound_[0]<=bound_[1]) {
            for (int i=bound_[2]; i<=bound_[3]; i++) {
                for (int j=bound_[0]; j<=bound_[1]; j++) {
                    float3 wi = cube2xyz(j, i, s, H);
                    float3 wh = safeNormalize(wi+normal);
                    float i_o_n = max(dot(wi,normal),0.0f);
                    float h_o_n = max(dot(wh,normal),0.0f);

                    float w = i_o_n * ndfGGX(alpha2,h_o_n)
                                    * pixel_area(j,i,H)/4.0f;
                    wsum += w;

                    for (int c=0; c<C; c++) {
                        col[c] += w*img[c*6*H*H+s*H*H+H*i+j];
                    }
                }
            }
        }
    }
    wsum = max(wsum,1e-8f);
    weight[0] = wsum;
    #pragma unroll
    for (int c=0; c<C; c++) {
        ret[c*H*H*6] = col[c]/= wsum;
    }
}

// gradient of cubemap filtering respect to feature
__global__ void specular_filtering_bwd_kernel(
    const int H, const int C, const float roughness,
    const float* grad, // Cx6xHxW
    const int* bound, // (6x4)x6xHxW
    const float* weight, //6xHxW
    float* ret // Cx6xHxW
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;

    if (px>=H || py>=H || pz>=6) {
        return;
    }

    float alpha2 = roughness*roughness;
    alpha2 = alpha2*alpha2;
    float3 normal = cube2xyz(px,py,pz,H);
    int idx = px + py*H + pz*H*H;
    grad += idx;
    bound += idx;
    weight += idx;

    float wsum = weight[0];
    float grad_in[16];
    #pragma unroll
    for (int c=0; c<C; c++){
        grad_in[c] =grad[c*6*H*H];
    }

    for (int s=0; s<6; s++) {
        int bound_[4];
        #pragma unroll 4
        for (int b=0;b<4;b++){
            bound_[b] = bound[b*6*H*H];
        }
        bound += 24*H*H;

        if (bound_[0]<=bound_[1]) {
            for (int i=bound_[2]; i<=bound_[3]; i++) {
                for (int j=bound_[0]; j<=bound_[1]; j++) {
                    float3 wi = cube2xyz(j, i, s, H);
                    float3 wh = safeNormalize(wi+normal);
                    float i_o_n = max(dot(wi,normal),0.0f);
                    float h_o_n = max(dot(wh,normal),0.0f);

                    float w = i_o_n * ndfGGX(alpha2,h_o_n)
                                    * pixel_area(j,i,H)/4.0f;

                    int offset = s*H*H+H*i+j;
                    #pragma unroll
                    for (int c=0; c<C; c++) {
                        atomicAdd(ret+(offset+c*6*H*H),w/wsum*grad_in[c]);
                    }
                }
            }
        }
    }
}






////////////////////////////////////////////////////////////////////////////
/**
    Range query a cubemap feature given direction and roughness
*/


// bilinear interpolate a specific mip level
template <int C>
__device__ void lerp_cube(
    float* img,
    float3 uvw, float r,
    int H,int T6,
    float* f
) {
    float u = clamp(uvw.x*(H-1),0.0f,float(H-1));
    float v = clamp(uvw.y*(H-1),0.0f,float(H-1));

    int u0 = int(floor(u));
    int u1 = int(ceil(u));
    int v0 = int(floor(v));
    int v1 = int(ceil(v));
    u = u-u0;
    v = v-v0;

    float _u = (1-u);
    float _v = (1-v);

    float w[4] = {
        _u*_v,_u*v,
        u*_v,u*v
    };

    int idx = u0*H+v0;
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] += w[0]*img[idx+c*T6]*r;
    }

    idx = u0*H+v1;
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] += w[1]*img[idx+c*T6]*r;
    }

    idx = u1*H+v0;
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] += w[2]*img[idx+c*T6]*r;
    }

    idx = u1*H+v1;
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] += w[3]*img[idx+c*T6]*r;
    }
}


// trilinear interpolate a cube map query
template <int C>
__global__ void fetch_mip_cube_kernel(
    int32_t B,
    float* img, //Cx6xN
    float* wi_, float* r_,
    int* offsets, 
    const int resolution, const int mip_level, int T,
    float* ret //CxB
) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b>=B) {
        return;
    }

    wi_ += b*3;
    r_ += b;
    ret += b;
    float3 wi = make_float3(wi_[0],wi_[1],wi_[2])*2-1;
    float3 uvw = xyz2cube(wi);
    float r = r_[0];
    r = clamp(r*(mip_level-1),0.0,float(mip_level-1));
    int r0 = int(floor(r));
    int r1 = int(ceil(r));
    r = r-r0;

    float f[C];
    #pragma unroll
    for (int c=0; c<C; c++) {
        f[c] = 0.0f;
    }
    int offset = offsets[r0] + int(uvw.z)*T;
    int H = (int)(resolution/exp2f(r0));

    lerp_cube<C>(img+offset, uvw, 1-r, H, T*6, f);
    if (r!=0.0f) {
        offset = offsets[r1] + int(uvw.z)*T;
        H = (int)(resolution/exp2f(r1));
        lerp_cube<C>(img+offset, uvw, r, H, T*6, f);
    }

    #pragma unroll
    for (int c=0; c<C; c++) {
        ret[B*c] = f[c];
    }
}


// gradient respect to uv and roughness (r)
template <int C>
__device__ float cube_duvdr(float* img, float* g, 
            float2 uv, float r,
            int H, int T6, 
            float* duv,float* img_grad) {
    float u = clamp(uv.x*(H-1),0.0f,float(H-1));
    float v = clamp(uv.y*(H-1),0.0f,float(H-1));

    int u0 = int(floor(u));
    int u1 = int(ceil(u));
    int v0 = int(floor(v));
    int v1 = int(ceil(v));
    u = u-u0;
    v = v-v0;

    float _u = (1-u);
    float _v = (1-v);
    
    int idx[4] = {u0*H+v0,u0*H+v1,u1*H+v0,u1*H+v1};
    float gf[4] = {0.0f,0.0f,0.0f};
    float w[4] = {
        _u*_v,_u*v,
        u*_v,u*v
    };
    
    #pragma unroll
    for (int k=0; k<4; k++) {
        #pragma unroll
        for (int c=0; c<C; c++) {
            gf[k] += g[c]*img[idx[k]+c*T6];
        }
    }
    if (r!=0.0f) {
        #pragma unroll
        for (int k=0; k<4; k++) {
            #pragma unroll
            for (int c=0; c<C; c++) {
                atomicAdd(img_grad+idx[k]+c*T6,w[k]*r*g[c]);
            }
        }
    }

    float dr = 0.0f;
    #pragma unroll
    for (int k=0; k<4; k++) {
        dr += gf[k]*w[k];
    }

    if (r!=0.0f) {
        // du
        w[0] = -_v;w[1] = -v;
        w[2] = _v;w[3] = v;
        #pragma unroll
        for (int k=0; k<4; k++) {
            duv[0] += w[k]*gf[k]*r*(H-1);
        }

        //dv
        w[0] = -_u;w[1] = _u;
        w[2] = -u;w[3] = u;
        #pragma unroll
        for (int k=0; k<4; k++) {
            duv[1] += w[k]*gf[k]*r*(H-1);
        }
    }

    return dr;
}

// gradient from cubemap uv to direction
__device__ float3 xyz2cube_bwd(int side, float3 wi, float* duv) {
    float3 ret;
    switch (side) {
        case 0:
            ret.x = -(wi.y*duv[0]+wi.z*duv[1])/(wi.x*wi.x);
            ret.y = duv[0]/wi.x;
            ret.z = duv[1]/wi.x;
            break;
        case 1:
            ret.y = -(wi.x*duv[0]+wi.z*duv[1])/(wi.y*wi.y);
            ret.x = duv[0]/wi.y;
            ret.z = duv[1]/wi.y;
            break; 
        case 2: 
            ret.z = -(wi.x*duv[0]+wi.y*duv[1])/(wi.z*wi.z);
            ret.x = duv[0]/wi.z;
            ret.y = duv[1]/wi.z;
            break;
        case 3:
            ret.x = (wi.y*duv[0]+wi.z*duv[1])/(wi.x*wi.x);
            ret.y = -duv[0]/wi.x;
            ret.z = -duv[1]/wi.x;
            break;
        case 4:
            ret.y = (wi.x*duv[0]+wi.z*duv[1])/(wi.y*wi.y);
            ret.x = -duv[0]/wi.y;
            ret.z = -duv[1]/wi.y;
            break; 
        case 5: 
            ret.z = (wi.x*duv[0]+wi.y*duv[1])/(wi.z*wi.z);
            ret.x = -duv[0]/wi.z;
            ret.y = -duv[1]/wi.z;
            break;
    }    
    return ret;
}


// gradient of cubemap query respect to feature, query direction, and roughness
template <int C>
__global__ void fetch_mip_cube_bwd_kernel(
    int32_t B,
    float* grad_in,
    float* img, //Cx6xT
    float* wi_, float* r_,
    int* offsets, 
    const int resolution, const int mip_level, int T,
    float* img_grad, //Cx6xT
    float* wi_grad, //Bx3
    float* r_grad //B
) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b>=B) {
        return;
    }

    grad_in += b;
    wi_ += b*3;
    r_ += b;
    wi_grad += b*3;
    r_grad += b;

    float g[C];
    #pragma unroll
    for (int c=0; c<C; c++) {
        g[c] = grad_in[c*B];
    }

    float3 wi = make_float3(wi_[0],wi_[1],wi_[2])*2-1;
    float3 uvw = xyz2cube(wi);
    float2 uv = make_float2(uvw.x,uvw.y);
    int side = int(uvw.z);
    float r = r_[0];
    r = clamp(r*(mip_level-1),0.0,float(mip_level-1));
    int r0 = int(floor(r));
    int r1 = int(ceil(r));
    r = r-r0;

    float duv[2] = {0.0f,0.0f};

    int offset = offsets[r0] + side*T;
    int H = (int)(resolution/exp2f(r0));
    float dr = -cube_duvdr<C>(img+offset, g, uv, 1-r, H, T*6, duv,img_grad+offset);
    
    offset = offsets[r1] + side*T;
    H = (int)(resolution/exp2f(r1));
    dr += cube_duvdr<C>(img+offset, g, uv, r, H, T*6, duv,img_grad+offset);

    uvw = xyz2cube_bwd(side, wi, duv);

    wi_grad[0] = uvw.x;
    wi_grad[1] = uvw.y;
    wi_grad[2] = uvw.z;
    r_grad[0] = dr*(mip_level-1);
}






////////////////////////////////////////////////////////////////////////////
// only support feature size of 3, 4, 8, 16

void specular_bound(
    const int resolution,
    const float cutoff,
    torch::Tensor ret
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dim3 threads = dim3(16,16,1);
    dim3 blocks = dim3(
        div_round_up<int32_t>(resolution,threads.x),
        div_round_up<int32_t>(resolution,threads.y),
        6
    );
    
    specular_bound_kernel<<<blocks,threads,0,stream>>>(
        resolution,cutoff,
        ret.data_ptr<int>());
}

void specular_filtering(
    torch::Tensor img,
    torch::Tensor bound,
    const float roughness,
    torch::Tensor ret,
    torch::Tensor weight
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t H = img.size(2);
    int32_t C = img.size(0);
    dim3 threads = dim3(16,16,1);
    dim3 blocks = dim3(
        div_round_up<int32_t>(H,threads.x),
        div_round_up<int32_t>(H,threads.y),
        6
    );
    
    specular_filtering_kernel<<<blocks,threads,0,stream>>>(
        H,C,roughness,
        img.data_ptr<float>(),
        bound.data_ptr<int>(),
        ret.data_ptr<float>(),
        weight.data_ptr<float>()
    );
}

void specular_filtering_bwd(
    torch::Tensor grad,
    torch::Tensor bound,
    torch::Tensor weight,
    const float roughness,
    torch::Tensor ret
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t H = ret.size(2);
    int32_t C = ret.size(0);
    dim3 threads = dim3(16,16,1);
    dim3 blocks = dim3(
        div_round_up<int32_t>(H,threads.x),
        div_round_up<int32_t>(H,threads.y),
        6
    );
    
    specular_filtering_bwd_kernel<<<blocks,threads,0,stream>>>(
        H,C,roughness,
        grad.data_ptr<float>(),
        bound.data_ptr<int>(),
        weight.data_ptr<float>(),
        ret.data_ptr<float>());
}

void fetch_mip_cube(
    torch::Tensor img,
    torch::Tensor wi,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    const int mip_level,
    int C,
    torch::Tensor ret
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t B = wi.size(0);
    int32_t T = img.size(2);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_cube_kernel<3><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            ret.data_ptr<float>());break;

        case 4: fetch_mip_cube_kernel<4><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            ret.data_ptr<float>());break;
        
        case 8: fetch_mip_cube_kernel<8><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            ret.data_ptr<float>());break;
        
        case 16: fetch_mip_cube_kernel<16><<<blocks,512,0,stream>>>(
            B,
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            ret.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature Cube: C must be 3, 4, 8, 16"};
    }
}


void fetch_mip_cube_bwd(
    torch::Tensor grad_in,
    torch::Tensor img, 
    torch::Tensor wi, torch::Tensor r,
    torch::Tensor offsets, 
    const int resolution, const int mip_level,
    int C,
    torch::Tensor img_grad,
    torch::Tensor wi_grad,
    torch::Tensor r_grad
) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t B = wi.size(0);
    int32_t T = img.size(2);
    dim3 blocks = dim3(div_round_up<int32_t>(B,512),1,1);
    switch (C) {
        case 3: fetch_mip_cube_bwd_kernel<3><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            img_grad.data_ptr<float>(),
            wi_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;

        case 4: fetch_mip_cube_bwd_kernel<4><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            img_grad.data_ptr<float>(),
            wi_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 8: fetch_mip_cube_bwd_kernel<8><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            img_grad.data_ptr<float>(),
            wi_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        
        case 16: fetch_mip_cube_bwd_kernel<16><<<blocks,512,0,stream>>>(
            B,
            grad_in.data_ptr<float>(),
            img.data_ptr<float>(),
            wi.data_ptr<float>(),
            r.data_ptr<float>(),
            offsets.data_ptr<int>(),
            resolution,mip_level,T,
            img_grad.data_ptr<float>(),
            wi_grad.data_ptr<float>(),
            r_grad.data_ptr<float>());break;
        default: throw std::runtime_error{"Feature Cube: C must be 3, 4, 8, 16"};
    }   
}