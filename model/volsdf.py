import math

import torch
import torch.nn as nn
import torch.nn.functional as NF

import numpy as np

import nerfacc

from .encoding import mlp, PositionalEncoding, HashEncoding
from .nde import NDE


""" volsdf network """
class SDFNet(nn.Module):
    def __init__(self,Cin,D=8,S=[4],C=256,weight_norm=False):
        super(SDFNet,self).__init__()
        self.D = D
        self.S = S
        self.C = C
        
        self.layer_act = nn.Softplus(beta=100)
        
        # intiialize network weight
        bias = 0.5
        linears = []
        for layer in range(self.D):
            if layer == 0:
                lin = nn.Linear(Cin,self.C)
            elif layer in self.S:
                lin = nn.Linear(self.C + Cin, self.C)
            else:
                lin = nn.Linear(self.C,self.C)
            
            if layer== 0:
                torch.nn.init.constant_(lin.bias,0.0)
                torch.nn.init.constant_(lin.weight[:,3:],0.0)
                torch.nn.init.normal_(lin.weight[:,:3],0.0,np.sqrt(2)
                                      /np.sqrt(lin.weight.shape[0]))
            elif layer in self.S:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.weight.shape[0]))
                torch.nn.init.constant_(lin.weight[:, -(Cin-3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias,0.0)
                torch.nn.init.normal_(lin.weight,0.0,np.sqrt(2)/np.sqrt(lin.weight.shape[0]))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            linears.append(lin)
        self.linears = nn.ModuleList(linears)
        
        linear_out = nn.Linear(self.C,1)
        torch.nn.init.normal_(linear_out.weight[0],mean=np.sqrt(np.pi)
                                      /np.sqrt(lin.weight.shape[1]),std=0.0001)
        torch.nn.init.constant_(linear_out.bias[0],-bias)
        if weight_norm:
            linear_out = nn.utils.weight_norm(linear_out)
        self.linear_out = linear_out

    def forward(self,x):
        f = x
        for i, l in enumerate(self.linears):
            if i in self.S:
                f = torch.cat([f,x], 1)
            f = l(f)
            f = self.layer_act(f)
        f = self.linear_out(f)
        return f







""" volsdf based specular nerf"""
class VolSDF(nn.Module):
    def __init__(self, aabb, sdf_config, nde_config):
        """
        Args:
            aabb: aabb boundxing box
            sdf_config: sdf config
            nde_config: nde config
        """
        super(VolSDF,self).__init__()
        xyzmin = aabb[:3]
        xyzmax = aabb[3:]
        self.register_buffer('xyzmin',torch.tensor(xyzmin).reshape(1,3))
        self.register_buffer('xyzmax',torch.tensor(xyzmax).reshape(1,3))
        
        # sdf network
        self.encode_sdf = PositionalEncoding(sdf_config.num_encode)
        self.mlp_sdf = SDFNet(self.encode_sdf.n_output_dims+3,D=sdf_config.D,C=sdf_config.C,S=sdf_config.S,weight_norm=sdf_config.weight_norm)
        self.beta = 0.1 # initial beta value
        
        # spatial network
        self.encode_x = HashEncoding(32,2048,16)
        self.mlp_x = mlp(self.encode_x.n_output_dims,[64],10+nde_config.Cx)
        
        # directional network
        self.nde = NDE(nde_config.Cx, nde_config.mlp_d, 
                            nde_config.far, nde_config.near, aabb, 
                            human=nde_config.get('human',None),pre_decode=nde_config.pre_decode)
    
    
    def get_density(self,sdf):
        """ convert sdf to volsdf density"""
        beta = self.beta
        alpha = 1.0/beta
        return alpha*(0.5+0.5*sdf.sign()*torch.expm1(-sdf.abs()/beta))
    
    def query_sdf(self,x):
        """ query sdf value"""
        x = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)*2-1
        sdf = self.mlp_sdf(torch.cat([x,self.encode_sdf(x*0.5+0.5).float()],dim=-1))[...,0]
        return sdf
    
    def query_density(self,x):
        """ qeury density value """
        x = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)*2-1
        mask = x.abs().max(-1)[0] <= 1
        sdf = self.mlp_sdf(torch.cat([x,self.encode_sdf(x*0.5+0.5).float()],dim=-1))[...,0]
        sigma = self.get_density(sdf)
        return sigma*mask
    
    
    def forward(self,x,d,estimator,human_pose=None,mode=0):
        """
        Args:
            x: Bx3 position
            d: Bx3 direction
            estimator: mip occupancy grid estimator
            human_pose: Bx3x4 capturer pose, None for no reflection of the capturer
            mode: nde display mode (0,1,2) for (all, far-field,near-field)
        Return:
            sigma: B density
            rgb: Bx3 color
            sdf: B sdf 
            sdf_grad: Bx3 sdf gradient
            normal: Bx3 surface normal
            
            roughness: B surface roughness
            ks: Bx3 specular tint
            Ls: specular color
            Ld: diffuse color
            
            sigma_n: near-field density at mip-level 0 (only returned in training)
        """
        
        # normalize x to [0,1]
        x0 = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)*2-1
        mask = x0.abs().max(-1)[0] <= 1
        x0 = x0.clone()
        
        # get sdf and sdf gradient
        x0.requires_grad = True
        sdf = self.mlp_sdf(torch.cat([x0,self.encode_sdf(x0*0.5+0.5).float()],dim=-1))[...,0]
        d_output = torch.ones_like(sdf,requires_grad=False)
        sdf_grad = torch.autograd.grad(
            outputs=sdf,
            inputs=x0,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        
        # convert sdf gradient to normal
        normal = NF.normalize(sdf_grad,eps=1e-6,dim=-1)
        
        # convert sdf to density
        sigma = self.get_density(sdf)
        sigma = sigma*mask
        
        # predict near-field density for regularization
        if self.training:
            sigma_n = self.nde.mlp_n_sigma(self.nde.encode_n_sigma(x0.detach()*0.5+0.5))[...,0]
            sigma_n = self.nde.sigma_act(sigma_n)
        
        # get spatial feature
        fx = self.mlp_x(self.encode_x(x0.detach()*0.5+0.5))
        roughness = fx[...,0].sigmoid()*0.99+0.01
        ks = fx[...,1:4].sigmoid()
        Ld = fx[...,4:7].sigmoid()
        #normal_pred = fx[...,7:10]
        #normal_pred = NF.normalize(normal_pred,dim=-1,eps=1e-6)
        fx = fx[...,10:]
        
        # reflected direction and cosine term
        wo = -d
        wo_o_n = (wo*normal).sum(-1,keepdim=True)
        wi = wo_o_n*2*normal-wo
        
        Ls = self.nde(x,wi,roughness,fx,wo_o_n,estimator,human_pose=human_pose,mode=mode)
        
        rgb = Ls*ks+Ld
        
        
        if self.training:
            return {
                'rgb': rgb,
                'sdf': sdf,
                'sigma_n': sigma_n,
                'sdf_grad': sdf_grad,
                'normal':normal,
                'sigma': sigma,
                
                'roughness': roughness,
                'ks': ks,
                'Ls': Ls,
                'Ld': Ld,
            }
        else:
            return {
                'rgb': rgb.detach(),
                'sdf': sdf.detach(),
                'sdf_grad': sdf_grad.detach(),
                'normal':normal.detach(),
                'sigma': sigma.detach(),
                
                'roughness': roughness.detach(),
                'ks': ks.detach(),
                'Ls': Ls.detach(),
                'Ld': Ld.detach()
            }
        