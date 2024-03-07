import math

import torch
import torch.nn as nn
import torch.nn.functional as NF
import tinycudann as tcnn

from .encoding import mlp, HashEncoding
    

    
""" hash grid with coordinate contraction for background modeling"""
class NeRF(nn.Module):
    def __init__(self,aabb):
        super(NeRF,self).__init__()
        xyzmin = aabb[:3]
        xyzmax = aabb[3:]
        self.register_buffer('xyzmin',torch.tensor(xyzmin).reshape(1,3))
        self.register_buffer('xyzmax',torch.tensor(xyzmax).reshape(1,3))
        self.sigma_act = NF.softplus
        
        
        max_resolution = 2048
        base_resolution = 16
        n_levels = 16
        per_level_scale = math.exp((math.log(max_resolution)-math.log(base_resolution))
                                 / (n_levels-1))
        n_features_per_level = 2
        log2_hashmap_size = 19
        self.encode_x = tcnn.Encoding(3,{
                            "otype": "HashGrid",
                            "n_levels": n_levels,
                            "n_features_per_level": n_features_per_level,
                            "log2_hashmap_size": log2_hashmap_size,
                            "base_resolution": base_resolution,
                            "per_level_scale": per_level_scale
                            })
        
        self.encode_d = tcnn.Encoding(3,{
            "otype": "SphericalHarmonics",
            "degree": 4})
        
        
        # mlp decoder
        self.mlp_x = mlp(self.encode_x.n_output_dims,[64],16+1)
        self.mlp_d = mlp(self.encode_d.n_output_dims+16,[64]*2,3)
        
    def contract(self,x):
        x = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)*2-1
        r = x.abs().max(-1)[0]
        valid = r>1
        x_out = torch.zeros_like(x)
        x_out[~valid] = x[~valid]
        r = r[valid].unsqueeze(-1).clamp_min(1e-4)
        x_out[valid] = (2-1/r)*(x[valid]/r)
        
        valid = torch.ones_like(valid)
        return x/4+0.5,valid
    
    def query_density(self,x):
        x,mask = self.contract(x)
        f = self.mlp_x(self.encode_x(x))
        return self.sigma_act(f[...,0])*mask
      
    def forward(self,x,d):
        x,mask = self.contract(x)
        f = self.mlp_x(self.encode_x(x))
        sigma = self.sigma_act(f[...,0])*mask
        f = f[...,1:]
        rgb = self.mlp_d(torch.cat([self.encode_d(d*0.5+0.5),f],-1)).sigmoid()

        return {
            'sigma': sigma,
            'rgb': rgb
        }