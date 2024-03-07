import math

import torch
import torch.nn as nn
import torch.nn.functional as NF

import tinycudann as tcnn # compile with fp32 for stable optimization

import sys
sys.path.append('..')
from utils.featuremap import specular_bound,specular_filtering,\
                             fetch_mip_cube,fetch_mip_2d,fetch_2d

""" input encoding models """





def mlp(dim_in,dims,dim_out):
    """ create a ReLU MLP in shape dim_in->dims[0]->...->dims[-1]->dim_out """
    mlps = [nn.Linear(dim_in,dims[0]),nn.ReLU()]
    for i in range(1,len(dims)):
        mlps.append(nn.Linear(dims[i-1],dims[i]))
        mlps.append(nn.ReLU())
    mlps.append(nn.Linear(dims[-1],dim_out))
    
    return nn.Sequential(*mlps)


""" positional encoding from NeRF """
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ L: number of frequency bands """
        super(PositionalEncoding, self).__init__()
        self.L= L
        self.n_output_dims = L*3*2
        
    def forward(self, inputs):
        L = self.L
        inputs = inputs*2-1
        encoded = []
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)
    
    

""" triplane enocding with mip-mapping support """
class TriplaneEncoding(nn.Module):
    def __init__(self,feature_size,res,level):
        """ feature_size: feature size
            res: triplane resolution
            level: number of mip level
        """
        super(TriplaneEncoding, self).__init__()
        
        self.feature_size = feature_size
        self.res = res
        self.level=level
        
        # get offsets of falttened mipmap
        offsets = [0]
        for i in range(0,self.level):
            offsets.append(offsets[-1]+(res//2**i)**2)
        offsets = torch.tensor(offsets,dtype=torch.long)
        self.register_buffer('offsets',offsets)
        
        self.n_output_dims = self.feature_size*3
        
        # intiialize triplane feature
        self.mat = nn.Parameter(0.1*torch.randn(3,self.feature_size,self.res,self.res))
        
    def create_mip(self,):
        """ create falttened mip map of the feature map """
        mat = self.mat
        mip_pool = torch.zeros(3,self.feature_size,self.offsets[-1],device=mat.device)
        mip_pool[:,:,:self.offsets[1]] = mat.reshape(3,self.feature_size,-1)

        for i in range(1,self.level):
            mat = NF.avg_pool2d(mat,2,2)
            mip_pool[:,:,self.offsets[i]:self.offsets[i+1]] = mat.reshape(3,self.feature_size,-1)
        
        return mip_pool
    
    def forward(self, x, r=None):
        """ 
        Args:
            x: Bx3 position in [0,1] 
            r: B query footprint, None for no mip-mapping
        Return:
            BxC queried feature
        """
        
        if r is None: # no mip-mapping
            f = torch.cat([
                fetch_2d(x[...,[0,1]],self.mat[0]).T,
                fetch_2d(x[...,[1,2]],self.mat[1]).T,
                fetch_2d(x[...,[2,0]],self.mat[2]).T,
            ],dim=-1)
            
        else: # range query given the footprint r
            if len(x) == 0:
                return torch.zeros(0,self.n_output_dims,device=x.device)
            
            mat_pool = self.create_mip() # create mip map
            r = torch.log2(2*r*self.res).clamp(0,self.level-1) # calculate mip level

            f = torch.cat([
                fetch_mip_2d(mat_pool[0],x[...,[0,1]],r,self.offsets.int(),self.res).T,
                fetch_mip_2d(mat_pool[1],x[...,[1,2]],r,self.offsets.int(),self.res).T,
                fetch_mip_2d(mat_pool[2],x[...,[2,0]],r,self.offsets.int(),self.res).T
            ],-1)
        return f
    

    
    
""" 2D plane enocding with mip-mapping support """
class PlaneEncoding(nn.Module):
    def __init__(self,feature_size,res,level):
        """ feature_size: feature size
            res: feature map resolution
            level: number of mip level
        """
        super(PlaneEncoding,self).__init__()
        self.feature_size = feature_size
        self.res = res
        self.level=level
        
        # get offsets of falttened mipmap
        offsets = [0]
        for i in range(0,self.level):
            offsets.append(offsets[-1]+(res//2**i)**2)
        offsets = torch.tensor(offsets,dtype=torch.long)
        self.register_buffer('offsets',offsets)
        
        self.n_output_dims = self.feature_size
        
        # intiialize plane feature
        self.mat = nn.Parameter(0.1*torch.randn(1,self.feature_size,self.res,self.res))
        
    def create_mip(self,):
        """ create falttened mip map of the feature map """
        mat = self.mat
        mip_pool = torch.zeros(self.feature_size,self.offsets[-1],device=mat.device)
        mip_pool[:,:self.offsets[1]] = mat.reshape(self.feature_size,-1)

        for i in range(1,self.level):
            mat = NF.avg_pool2d(mat,2,2)
            mip_pool[:,self.offsets[i]:self.offsets[i+1]] = mat.reshape(self.feature_size,-1)
        return mip_pool
    
    def forward(self, x,r=None):
        """ 
        Args:
            x: Bx2 position in [0,1] 
            r: B query footprint, None for no mip-mapping
        Return:
            BxC querfied feature
        """
        
        if r is None: # no mip-mapping
            f = fetch_2d(x,self.mat).T
        else: # range query given the footprint r
            if len(x) == 0:
                return torch.zeros(0,self.n_output_dims,device=x.device)
            
            mat_pool = self.create_mip() # create mip map
            r = torch.log2(2*r*self.res).clamp(0,self.level-1) # calculate mip level

            f = fetch_mip_2d(mat_pool,x,r,self.offsets.int(),self.res).T
        return f
    
    

""" cubemap enocding with mip-mapping support """    
class CubemapEncoding(nn.Module):
    def __init__(self,feature_size,res,level):
        """ feature_size: feature size
            res: feature map resolution
            level: number of mip level
        """
        super(CubemapEncoding,self).__init__()
        self.res = res
        self.level = level
        self.feature_size = feature_size
        self.n_output_dims = feature_size
        
        # initialize feature map
        feature = torch.randn(feature_size,6,res,res)*0.1
        self.register_parameter('feature',nn.Parameter(feature))
        
        # get offsets of falttened mipmap
        offsets = torch.tensor([int((res//2**i)**2) for i in range(level)]).cumsum(0)
        offsets = torch.cat([torch.zeros(1,),offsets],0)
        self.register_buffer('offsets',offsets.long())
        
        # initialize filtering bounds for each sampled roughness value
        self.register_buffer('roughness',torch.linspace(0.03,0.99,self.level))
        for i in range(self.level):
            bound = specular_bound(self.res//2**i,
                                   self.roughness[i],0.9,torch.device(0))
            self.register_buffer('bound{}'.format(i),bound.cpu())
        torch.cuda.empty_cache()
        
    def create_mip(self,):
        """ create falttened mip map of the feature map """
        mip_pool = torch.zeros(self.feature_size,6,self.offsets[-1],device=self.offsets.device)
        feature = self.feature
        mip_pool[...,self.offsets[0]:self.offsets[1]] = feature.reshape(self.feature_size,6,-1)
        
        for i in range(1,self.level):
            feature = NF.avg_pool2d(feature,2,2)
            feature_filtered = specular_filtering(feature,getattr(self,'bound{}'.format(i)),self.roughness[i])
            mip_pool[...,self.offsets[i]:self.offsets[i+1]] = feature_filtered.reshape(self.feature_size,6,-1)
        
        return mip_pool
    
    def forward(self,w,r):
        """ 
        Args:
            w: Bx3 query direction in [0,1] (w*0.5+0.5)
            r: B roughness 
        Return:
            BxC queried feature
        """
        
        mip_pool = self.create_mip() # create mipmap
        ret = fetch_mip_cube(mip_pool,w,r,self.offsets.int(),self.res,self.level)
        
        return ret.T
    
    
    
""" create a hash grid encoding"""
def HashEncoding(feature_size=32,res=2048,level=16):
    """ feature_size: feature size
        res: feature map resolution
        level: number of mip level
    """
    max_resolution = res
    base_resolution = 16
    n_levels = level
    per_level_scale = math.exp((math.log(max_resolution)-math.log(base_resolution))
                                 / (n_levels-1))
    n_features_per_level = feature_size//n_levels
    log2_hashmap_size = 19
    
    return tcnn.Encoding(3,{
        "otype": "HashGrid",
        "n_levels": n_levels,
        "n_features_per_level": n_features_per_level,
        "log2_hashmap_size": log2_hashmap_size,
        "base_resolution": base_resolution,
        "per_level_scale": per_level_scale
    })