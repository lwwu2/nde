import torch
import nerfacc


def gamma(x):
    """ tone mapping function """
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92*x[mask]
    mask = ~mask
    ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
    return ret



def volume_rendering(rays_x, rays_d, near, far, render_step_size, rgb_bkgd,
                     model, estimator, estimator2,
                     human_pose=None, mode=0,shape_loss=False):
    """ volume rendering the foreground
    Args:
        rays_x: Bx3 ray origin
        rays_d: Bx3 ray direction
        near,far: near and far plane
        render_step_size: sampling step size
        rgb_bkgd: (Bx3,3) background color
        model: NeRF model
        estimator: occupancy grid estimator
        estimator2: mip occupancy grid estimator
        human_pose: Bx3x4 capturer pose, None of no capturer
        mode: (0,1,2) for (all,far-field,near-field) reflection
        shape_loss: whether calculate shape loss from NeRO
    """
    
    # primary ray sampling
    def sigma_fn(t_starts, t_ends, ray_indices):
        x = rays_x[ray_indices]\
          + 0.5*(t_starts+t_ends).unsqueeze(-1)*rays_d[ray_indices]
        with torch.no_grad():
             sigma = model.query_density(x)
        return sigma
    ray_indices, t_starts, t_ends = estimator.sampling(
        rays_x,rays_d,
        sigma_fn=sigma_fn,
        t_min=near,t_max=far,
        render_step_size=render_step_size,
        stratified=True,
        alpha_thre=0.0
    )
    
    n_rays = len(rays_x)
    n_rendering_samples = len(t_starts)
    if n_rendering_samples == 0:
        return None
    
    # network query
    positions = rays_x[ray_indices] + ((t_starts+t_ends)*0.5).unsqueeze(-1)*rays_d[ray_indices]
    if human_pose is not None:
        human_pose = human_pose[ray_indices]
    with torch.set_grad_enabled(True):
        ret = model(positions,rays_d[ray_indices],estimator2,human_pose=human_pose,mode=mode)

    
    # volume rendering
    def rgb_sigma_fn(t_starts,t_ernds,ray_indices):
        sigma_est = ret['sigma']
        rgb_est = ret['rgb']
        return rgb_est,sigma_est
    rgbs,_,depths,extras = nerfacc.rendering(
        t_starts,t_ends,
        ray_indices,
        n_rays=n_rays,
        rgb_sigma_fn=rgb_sigma_fn,
        render_bkgd=rgb_bkgd
    )
    
    
    weights = extras['weights']
    if model.training == False:
        # accumulate additional attributes
        normals = nerfacc.accumulate_along_rays(weights,ret['normal']*0.5+0.5,
                            ray_indices=ray_indices,n_rays=n_rays)
        roughness = nerfacc.accumulate_along_rays(weights,
                            ret['roughness'].reshape(-1,1),
                            ray_indices=ray_indices,n_rays=n_rays)
        Ls = nerfacc.accumulate_along_rays(weights,
                            ret['Ls'],
                            ray_indices=ray_indices,n_rays=n_rays)
        Ld = nerfacc.accumulate_along_rays(weights,
                            ret['Ld'],
                            ray_indices=ray_indices,n_rays=n_rays)
        return {
            'rgb': rgbs,
            'depth': depths,
            'normal': normals,
            'roughness': roughness,
            'Ls': Ls,
            'Ld': Ld
        }
    else:
        # near-field density regularization
        def rgb_sigma_fn_pred(t_starts,t_ernds,ray_indices):
            sigma_est = ret['sigma_n']
            rgb_est = ret['rgb'].detach()
            return rgb_est,sigma_est
        rgbs_n,_,_,_ = nerfacc.rendering(
            t_starts,t_ends,
            ray_indices,
            n_rays=n_rays,
            rgb_sigma_fn=rgb_sigma_fn_pred,
            render_bkgd=rgb_bkgd
        )
        
        # shape stabilization loss
        if shape_loss:
            small_threshold = 0.1*(model.xyzmax[0,0]-model.xyzmin[0,0])
            large_threshold = 1.05*(model.xyzmax[0,0]-model.xyzmin[0,0])
            points = positions
                
            sdf = model.query_sdf(points)
            r = points.norm(dim=-1)
            small_mask = r<small_threshold
            loss_shape = torch.zeros(len(points),device=points.device)
            if small_mask.any():
                bounds = r[small_mask]-small_threshold
                loss_shape[small_mask] = (sdf[small_mask]-bounds).relu()
                
            large_mask = r > large_threshold
            if large_mask.any():
                bounds = r[large_mask]-large_threshold
                loss_shape[large_mask] = (bounds-sdf[large_mask]).relu()
            loss_shape = loss_shape.mean()
        else:
            loss_shape = 0.0
        
        return {
            'rgb': rgbs,
            'rgb_n': rgbs_n,
            'sdf_grad': ret['sdf_grad'],
            'loss_shape': loss_shape,
            'n_rendering_samples': n_rendering_samples
        }

    
def volume_rendering_bkgd(rays_x,rays_d,tmaxs,render_step_size_bkgd,
                         model_bkgd,estimator3):
    """ volume rendering the background
    Args:
        rays_x: Bx3 ray origin
        rays_d: Bx3 ray direction
        tmaxs: foreground far plane
        render_step_size_bkgd: background sampling step size
        model_bkgd: background nerf model
        estimator3: background occupancy grid estimator
    """
    # step out of the foreground
    rays_x = rays_x + rays_d*tmaxs.reshape(-1,1)*1.03
    
    
    # occupancy grid sampling the point
    def sigma_fn3(t_starts, t_ends, ray_indices):
        x = rays_x[ray_indices]\
          + 0.5*(t_starts+t_ends).unsqueeze(-1)*rays_d[ray_indices]
        with torch.no_grad():
             sigma = model_bkgd.query_density(x)
        return sigma
    
    ray_indices, t_starts, t_ends = estimator3.sampling(
        rays_x,rays_d,
        sigma_fn=sigma_fn3,
        near_plane=0.01,far_plane=1e10,
        render_step_size=render_step_size_bkgd,
        stratified=True,
        alpha_thre=0.0
    )
    n_rays = len(rays_x)
    n_rendering_samples = len(t_starts)
    if n_rendering_samples == 0:
        return torch.zeros_like(rays_x),False
    
    # query the nerf
    ray_dist = (t_starts+t_ends)*0.5
    positions = rays_x[ray_indices] + ray_dist.unsqueeze(-1)*rays_d[ray_indices]
    ret = model_bkgd(positions,rays_d[ray_indices])
    
    # augmentation by random background color
    bkgd = torch.rand(3,device=rays_x.device)
    def rgb_sigma_fn(t_starts,t_ernds,ray_indices):
        sigma_est = ret['sigma']
        rgb_est = ret['rgb']
        return rgb_est,sigma_est
    rgbs,_,_,extras = nerfacc.rendering(
        t_starts,t_ends,
        ray_indices,
        n_rays=n_rays,
        rgb_sigma_fn=rgb_sigma_fn,
        render_bkgd=bkgd
    )

    return rgbs,True