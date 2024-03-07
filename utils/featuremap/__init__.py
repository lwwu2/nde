import torch
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_bwd, custom_fwd
import math
from pathlib import Path


_ext_src_root = Path(__file__).parent / 'src'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_ext = load(name='featuremap_ext', 
            sources=_ext_src_files,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths)


""" cubemap filtering """

def specular_bound(res,roughness,cutoff,device):
    """ find the aabb bound for cubemap filtering of a GGX kernel
    Args:
        res: cubemap resolution
        roughness: surface roughness
        cutoff: cutoff contribution
        device: cuda device
    Return:
        6x4x6xresxres aabb bound
    """
    alpha = float(roughness*roughness)
    cutoff = math.sqrt((1-cutoff)/(cutoff*(alpha*alpha-1)+1))
    
    ret = torch.zeros(6,4,6,res,res,dtype=torch.int32,device=device)
    
    _ext.specular_bound(res,cutoff,ret.contiguous())
    return ret

class SpecularFiltering(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,img,bound,roughness):
        """ filtering cubemap by a GGX kernel
        Args:
            img: 6xCxHxH cubemap features
            bound: 6x4xHxH aabb filtering bound
            roughness: roughness for the GGX kernel
        Return:
            Cx6xHxH filtered feature map
        """
        C,_,H,_ = img.shape
        assert C<=16 # only supports less than 16 features
        roughness = float(roughness)
        
        ret = torch.zeros(C,6,H,H,device=img.device)
        weight = torch.zeros(6,H,H,device=img.device) # save for backward gradient computation
        
        _ext.specular_filtering(img.contiguous(),bound.contiguous(),roughness,
                               ret.contiguous(),weight.contiguous())
        
        ctx.save_for_backward(weight,bound)
        ctx.roughness = roughness
        ctx.feature_size = C
        return ret
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        """
        Args:
            Cx6xHxH input gradient
        Return:
            Cx6xHxH feature map gradient
        """
        weight,bound = ctx.saved_tensors
        H = bound.shape[-1]
        C = ctx.feature_size
        
        ret = torch.zeros(C,6,H,H,device=bound.device)
        
        _ext.specular_filtering_bwd(grad_in.contiguous(),bound.contiguous(),
                                    weight.contiguous(),ctx.roughness,
                                    ret.contiguous())
        return ret,None,None

specular_filtering = SpecularFiltering.apply




""" feature map query """

class _fetch_mip_cube(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,img,wi,rho,offsets,res,mip_level):
        """ trilinear interpolation of a cube feature map 
        Args:
            img: CX6xT flattened L-level cubamap featrues, T=(H*H+H/2*H/2+...+H/2**L*H/2**L)
            wi: Bx3 query direction
            rho: B query roughness (mip level)
            offsets: L+1 offset of each mip level
            res: finest feature map resolution
            mip_level: number of mip levels
        Return:
            CxB queried features
        """
        C = img.shape[0]
        B = wi.shape[0]
        assert C<=16  
        
        ret = torch.zeros(C,B,device=wi.device)
        
        _ext.fetch_mip_cube(img.contiguous(),wi.contiguous(),rho.contiguous(),
                       offsets.contiguous(),res,mip_level,C,
                       ret.contiguous())
        
        ctx.save_for_backward(img,wi,rho,offsets)
        ctx.res = res
        ctx.mip_level = mip_level
        return ret
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        """
        Args:
            grad_in: CxB input gradient
        Returns:
            img_grad: Cx6xT output cubemap feature gradient
            wi_grad: Bx3 output direction gradient
            rho_grad: B output roughness gradient
        """
        img,wi,rho,offsets = ctx.saved_tensors
        C = img.shape[0]
        res = ctx.res
        mip_level = ctx.mip_level
        
        img_grad = torch.zeros(img.shape,device=img.device)
        wi_grad = torch.zeros(wi.shape,device=wi.device)
        rho_grad = torch.zeros(rho.shape,device=rho.device)
        
        _ext.fetch_mip_cube_bwd(grad_in.contiguous(),
                          img.contiguous(),wi.contiguous(),rho.contiguous(),
                           offsets.contiguous(),res,mip_level,C,
                           img_grad.contiguous(),wi_grad.contiguous(),rho_grad.contiguous())
        
        return img_grad, wi_grad, rho_grad, None,None,None
    
fetch_mip_cube = _fetch_mip_cube.apply



class _fetch_mip_2d(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,img,xy,r,offsets,res):
        """ trilinear interpolation of a 2D feature map 
        Args:
            img: CxT flattened L-level 2D featrues, T=(H*H+H/2*H/2+...+H/2**L*H/2**L)
            xy: Bx2 query position
            r: B query mip level
            offsets: L+1 offset of each mip level
            res: finest feature map resolution
        Return:
            CxB queried features
        """
        C = img.shape[0]
        
        ret = torch.zeros(C,xy.shape[0],device=xy.device)
        
        _ext.fetch_mip_2d(img.contiguous(),xy.contiguous(),r.contiguous(),
                         offsets.contiguous(),res,
                         ret.contiguous())
        
        ctx.save_for_backward(img,xy,r,offsets)
        ctx.res = res
        return ret
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        """
        Args:
            grad_in: CxB input gradient
        Return:
            img_grad: CxT output image gradient
            xy_grad: Bx2 output position gradient
            r_grad: B output mip level gradient
        """
        img,xy,r,offsets = ctx.saved_tensors
        C = img.shape[0]
        res = ctx.res
        
        img_grad = torch.zeros(img.shape,device=img.device)
        xy_grad = torch.zeros(xy.shape,device=xy.device)
        r_grad = torch.zeros(r.shape,device=r.device)
        
        _ext.fetch_mip_2d_bwd(grad_in.contiguous(),
                          img.contiguous(),xy.contiguous(),r.contiguous(),
                           offsets.contiguous(),res,
                           img_grad.contiguous(),xy_grad.contiguous(),r_grad.contiguous())
        
        return img_grad, xy_grad, r_grad, None,None,None

fetch_mip_2d = _fetch_mip_2d.apply


class _fetch_mip_1d(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,img,x,r,offsets,res):
        """ trilinear interpolation of a 1D feature map 
        Args:
            img: CxT flattened L-level 1D featrues, T=(H+H/2+...+H/2**L)
            x: B query position
            r: B query mip level
            offsets: L+1 offset of each mip level
            res: finest feature map resolution
        Return:
            CxB queried features
        """
        C = img.shape[0]
        
        ret = torch.zeros(C,x.shape[0],device=x.device)
        
        _ext.fetch_mip_1d(img.contiguous(),x.contiguous(),r.contiguous(),
                         offsets.contiguous(),res,
                         ret.contiguous())
        
        ctx.save_for_backward(img,x,r,offsets)
        ctx.res = res
        return ret
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_in):
        """
        Args:
            grad_in: CxB input gradient
        Return:
            img_grad: CxT output image gradient
            x_grad: B output position gradient
            r_grad: B output mip level gradient
        """
        img,x,r,offsets = ctx.saved_tensors
        C = img.shape[0]
        res = ctx.res
        
        img_grad = torch.zeros(img.shape,device=img.device)
        x_grad = torch.zeros(x.shape,device=x.device)
        r_grad = torch.zeros(r.shape,device=r.device)
        
        _ext.fetch_mip_1d_bwd(grad_in.contiguous(),
                          img.contiguous(),x.contiguous(),r.contiguous(),
                           offsets.contiguous(),res,
                           img_grad.contiguous(),x_grad.contiguous(),r_grad.contiguous())
        
        return img_grad, x_grad, r_grad, None,None,None

fetch_mip_1d = _fetch_mip_1d.apply



def fetch_2d(xy,f):
    """ bilinear interpolate 2d feature map without mip-mapping
    Args:
        xy: Bx2 position
        f: CxHxH feature map
    Return:
        CxB feature map
    """
    C,H,_ = f.shape
    xy = xy.mul(H-1).clamp(0,H-1)
    
    xy0 = xy.floor().long()
    xy1 = xy.ceil().long()
    xy = xy-xy0
    x0,y0 = xy0[...,0],xy0[...,1]
    x1,y1 = xy1[...,0],xy1[...,1]
    x,y = xy[...,0],xy[...,1]
    
    fout = f[:,y0,x0]*(1-x)*(1-y) + f[:,y0,x1]*x*(1-y)\
      + f[:,y1,x0]*(1-x)*y + f[:,y1,x1]*x*y
    
    return fout