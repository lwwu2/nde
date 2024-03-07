import math

import torch
import torch.nn as nn
import torch.nn.functional as NF

import nerfacc

from .encoding import mlp, TriplaneEncoding, PlaneEncoding, CubemapEncoding



def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    from  https://github.com/liuyuan-pal/NeRO/blob/main/network/field.py
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:,:,:3], poses[:,:,3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:,:,None] + t)[..., 0] # pn,3
    dirs_ = (R @ dirs[:,:,None])[..., 0]   # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    hits &= dist>0
    
    
    return inter[...,:2], dist, hits




""" neural directional encoding"""
class NDE(nn.Module):
    def __init__(self,Cx, mlp_d,far,near,aabb,human=None,pre_decode=False):
        """ 
        Args:
            Cx: spatial feature size
            mlp_d: decoder mlp config
            far: far-field feature config
            near: near-field feature config
            aabb: aabb bounding box of the scene
            human: human feature config, None for no human feature
            pre_decode: whether decoding then blend the feature
        """
        super(NDE,self).__init__()
        # scene bounding box
        xyzmin = aabb[:3]
        xyzmax = aabb[3:]
        self.register_buffer('xyzmin',torch.tensor(xyzmin).reshape(1,3))
        self.register_buffer('xyzmax',torch.tensor(xyzmax).reshape(1,3))
        
        # far-field feature
        self.encode_f = CubemapEncoding(far.C,far.H,far.L) # 16,64,6
        
        
        # near-field feature + decoder mlp
        self.encode_n = TriplaneEncoding(near.C,near.H,near.L) # 16,512,9
        self.encode_n_sigma = TriplaneEncoding(near.C,near.H,near.L) # 16,512,9
        
        self.mlp_n = mlp(self.encode_n.n_output_dims,
                         near.mlp,far.C) # [64], 16
        self.mlp_n_sigma = mlp(self.encode_n_sigma.n_output_dims,
                         near.mlp,1) # [64]
        
        self.sigma_act = NF.softplus
        
        
        # human feature + decoder mlp
        if human is not None:
            self.encode_h = PlaneEncoding(human.C,human.H,human.L) # [16,256,8]
            self.mlp_h = mlp(self.encode_h.n_output_dims,human.mlp,far.C+1)
        
        # color decoder mlp
        self.mlp_d = mlp(Cx+self.encode_f.n_output_dims+1, mlp_d,3) # 16+16+1,[64]*2
        

        
        # initial sampling step size of near-field feature
        self.render_step_size = 5e-2
        self.pre_decode = pre_decode
        self.scale_human = 0.1 # size of the human feature plane
        
        
    def query_n_density(self,x,r):
        """ query near-field feature density """
        # normalize to [0,1]
        x = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)
        r = r/(self.xyzmax-self.xyzmin)[...,-1]
        valid = (x>=0).all(-1)&(x<=1).all(-1)
        
        f = self.mlp_n_sigma(self.encode_n_sigma(x,r))
        sigma = self.sigma_act(f.squeeze(-1))
        return sigma*valid
        
    def query_n(self,x,r):
        """ query near-field feature and density """
        x = (x-self.xyzmin)/(self.xyzmax-self.xyzmin)
        r = r/(self.xyzmax-self.xyzmin)[...,-1]
        valid = (x>=0).all(-1)&(x<=1).all(-1)
        
        f = self.mlp_n_sigma(self.encode_n_sigma(x,r))
        sigma = self.sigma_act(f.squeeze(-1))
        h_n = self.mlp_n(self.encode_n(x,r))
        return sigma*valid,h_n
    
    def forward(self,x,wi,roughness,fx,wo_o_n,estimator,human_pose=None,mode=0):
        """
        Args:
            x: Bx3 position
            wi: Bx3 reflected direction
            roughness: B surface roughness
            fx: BxC spatial feature
            wo_o_n: Bx1 cosine term
            estimator: mip occupancy grid estimator
            human_pose: Bx3x4 human pose for reflection of the captuerer, None for no captuerer reflection
            mode: feature mode, 0: all, 1: far-field, 2: near-field
        Return:
            Bx3 view-dependent color
        """
        
        # query far-field feature
        h_f = self.encode_f(wi*0.5+0.5,roughness)
        
        
        
        # cone trace the near-field feature
        t_min,t_max = 1e-2,2.0 # min and max marching step
        
        # find cone radius, 75% contribution of a GGX lobe
        T = 0.75
        r0 = roughness.pow(2)*math.sqrt(T/(1-T))
        
        # sampling a mip-mapped occupancy grid
        def sigma_fn(t_starts, t_ends, ray_indices):
            with torch.no_grad():
                t_ = (t_starts+t_ends)*0.5
                p = x[ray_indices]\
                  + t_.unsqueeze(-1)*wi[ray_indices]
                r = r0[ray_indices]*t_
                sigma = self.query_n_density(p,r)
            return sigma
        ray_indices,t_starts,t_ends = estimator.sampling(
            x,wi,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=sigma_fn,
            render_step_size=self.render_step_size,
            early_stop_eps=1e-2,
            alpha_thre=0.0,
            stratified=True,
            cone_angle=r0.data,
        )
        t_ = 0.5*(t_starts+t_ends)
        x_n = x[ray_indices]+t_.unsqueeze(-1)*wi[ray_indices]
        r_n = r0[ray_indices]*t_
        
        
        # sample near-field features
        sigma_n,h_n = self.query_n(x_n,r_n)
        if self.pre_decode: # pre-decode the color
            h_n = self.mlp_d(torch.cat([fx[ray_indices],h_n,wo_o_n[ray_indices]],-1)).sigmoid()
        
        # accumulate near-field features along the ray
        n_rays = len(x)
        weights_n,_,_ = nerfacc.render_weight_from_density(t_starts,t_ends, 
                                        sigma_n,ray_indices=ray_indices,n_rays=n_rays)
        h_n = nerfacc.accumulate_along_rays(weights_n,h_n,
                                          ray_indices=ray_indices,n_rays=n_rays)
        alpha_n = nerfacc.accumulate_along_rays(weights_n, None, ray_indices, n_rays)
        
        
        # accumulate captuerer feature
        if human_pose is not None:         
            # find ray-plane intersection
            uv_human,t_human,hits = get_camera_plane_intersection(x, wi, human_pose)
            uv_human = (uv_human*self.scale_human)*0.5+0.5
            t_human = (t_human*self.scale_human)*0.5
            hits &= (uv_human>=0).all(-1)&(uv_human<=1).all(-1)
            
            # detach the footprint gradient
            r_human = r0.detach()*t_human
            
            # decode the captuerer feature
            h_human = torch.zeros_like(h_n)
            alpha_human = torch.zeros_like(alpha_n)
            if len(r_human)>0:
                h_human_ = self.mlp_h(self.encode_h(uv_human[hits],r_human[hits]))
                alpha_human[hits] = 1-torch.exp(-self.sigma_act(h_human_[...,0:1]))
                
                if self.pre_decode:
                    h_human[hits] = self.mlp_d(torch.cat([fx[hits],h_human_[...,1:],wo_o_n[hits]],-1)).sigmoid()
                else:
                    h_human[hits] = h_human_[...,1:]
                
                h_n = h_n + alpha_human*h_human
                alpha_n = alpha_n + (1-alpha_n)*alpha_human

                
                
                
                
                
        # blend near- and far-field features
        if self.pre_decode: # the feature has been decoded
            Ls = self.mlp_d(torch.cat([fx,h_f,wo_o_n],-1)).sigmoid()
            
            if mode == 0:
                Ls = Ls*(1-alpha_n) + h_n
            elif mode == 2:
                Ls = h_n
        else:
            if mode == 0:
                h = h_f*(1-alpha_n) + h_n
            elif mode == 1:
                h = h_f
            elif mode == 2:
                h = h_n

            Ls = self.mlp_d(torch.cat([fx,h,wo_o_n],-1)).sigmoid()
        
        return Ls
        