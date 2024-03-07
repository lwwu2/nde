"""
The folder is modified from the nerfacc codebase: https://github.com/nerfstudio-project/nerfacc/tree/master/nerfacc/cuda
"""

import torch
from torch import Tensor
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_bwd, custom_fwd
import math
from pathlib import Path

from typing import Any, Callable, Optional, Tuple, List, Union


import nerfacc
from nerfacc.data_specs import RayIntervals, RaySamples


_ext_src_root = Path(__file__).parent / 'src'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_ext = load(name='traverse_mip_grids_ext', 
            sources=_ext_src_files,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths)


"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
@torch.no_grad()
def traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    binaries: Tensor,  # [m, resx, resy, resz]
    aabbs: Tensor,  # [m, 6]
    # options
    near_planes: Optional[Tensor] = None,  # [n_rays]
    far_planes: Optional[Tensor] = None,  # [n_rays]
    step_size: Optional[float] = 1e-3,
    cone_angle: Optional[Tensor]=None,
    traverse_steps_limit: Optional[int] = None,
    over_allocate: Optional[bool] = False,
    rays_mask: Optional[Tensor] = None,  # [n_rays]
    # pre-compute intersections
    t_sorted: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    t_indices: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    hits: Optional[Tensor] = None,  # [n_rays, n_grids]
) -> Tuple[RayIntervals, RaySamples, Tensor]:
    """Ray Traversal within Multiple Grids.

    Note:
        This function is not differentiable to any inputs.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        binary_grids: (m, resx, resy, resz) Multiple binary grids with the same resolution.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_planes: Optional. (n_rays,) Near planes for the traversal to start. Default to 0.
        far_planes: Optional. (n_rays,) Far planes for the traversal to end. Default to infinity.
        step_size: Optional. Step size for ray traversal. Default to 1e-3.
        cone_angle: Optional. Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.
        traverse_steps_limit: Optional. Maximum number of samples per ray.
        over_allocate: Optional. Whether to over-allocate the memory for the outputs.
        rays_mask: Optional. (n_rays,) Skip some rays if given.
        t_sorted: Optional. (n_rays, n_grids * 2) Pre-computed sorted t values for each ray-grid pair. Default to None.
        t_indices: Optional. (n_rays, n_grids * 2) Pre-computed sorted t indices for each ray-grid pair. Default to None.
        hits: Optional. (n_rays, n_grids) Pre-computed hit flags for each ray-grid pair. Default to None.

    Returns:
        A :class:`RayIntervals` object containing the intervals of the ray traversal, and
        a :class:`RaySamples` object containing the samples within each interval.
        t :class:`Tensor` of shape (n_rays,) containing the terminated t values for each ray.
    """
    if cone_angle is None:
        cone_angle = torch.zeros_like(rays_o[:,0])
    if near_planes is None:
        near_planes = torch.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = torch.full_like(rays_o[:, 0], float("inf"))

    if rays_mask is None:
        rays_mask = torch.ones_like(rays_o[:, 0], dtype=torch.bool)
    if traverse_steps_limit is None:
        traverse_steps_limit = -1
    if over_allocate:
        assert (
            traverse_steps_limit > 0
        ), "traverse_steps_limit must be set if over_allocate is True."

    if t_sorted is None or t_indices is None or hits is None:
        # Compute ray aabb intersection for all levels of grid. [n_rays, m]
        t_mins, t_maxs, hits = nerfacc.ray_aabb_intersect(rays_o, rays_d, aabbs[0:1])
        # Sort the t values for each ray. [n_rays, m]
        t_sorted, t_indices = torch.sort(
            torch.cat([t_mins, t_maxs], dim=-1), dim=-1
        )

    # Traverse the grids.
    intervals, samples, termination_planes = _ext.traverse_grids(
        # rays
        rays_o.contiguous(),  # [n_rays, 3]
        rays_d.contiguous(),  # [n_rays, 3]
        rays_mask.contiguous(),  # [n_rays]
        # grids
        binaries.contiguous(),  # [m, resx, resy, resz]
        aabbs.contiguous(),  # [m, 6]
        # intersections
        t_sorted.contiguous(),  # [n_rays, m * 2]
        t_indices.contiguous(),  # [n_rays, m * 2]
        hits.contiguous(),  # [n_rays, m]
        # options
        near_planes.contiguous(),  # [n_rays]
        far_planes.contiguous(),  # [n_rays]
        step_size,
        cone_angle.contiguous(), # [n_rays]
        True,
        True,
        True,
        traverse_steps_limit,
        over_allocate,
    )
    return (
        RayIntervals._from_cpp(intervals),
        RaySamples._from_cpp(samples),
        termination_planes,
    )





import torch.nn.functional as NF
from nerfacc import  (
    render_visibility_from_alpha,
    render_visibility_from_density,
)
from nerfacc.estimators.occ_grid import _meshgrid3d





""" The occupancy grid estimator that querieds a mip-mapped occupancy grid"""
class MIPOccGridEstimator(nerfacc.estimators.base.AbstractEstimator):
    DIM: int = 3
    def __init__(
        self,
        roi_aabb: Union[List[int], Tensor],
        resolution: Union[int, List[int], Tensor] = 128,
        levels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if "contraction_type" in kwargs:
            raise ValueError(
                "`contraction_type` is not supported anymore for nerfacc >= 0.4.0."
            )

        # check the resolution is legal
        if isinstance(resolution, int):
            resolution = [resolution] * self.DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, Tensor), f"Invalid type: {resolution}!"
        assert resolution.shape[0] == self.DIM, f"Invalid shape: {resolution}!"

        # check the roi_aabb is legal
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, Tensor), f"Invalid type: {roi_aabb}!"
        assert roi_aabb.shape[0] == self.DIM * 2, f"Invalid shape: {roi_aabb}!"

        # multiple levels of aabbs
        aabbs = roi_aabb.unsqueeze(0)
        

        # total number of voxels
        self.cells_per_lvl = int(resolution.prod().item())
        self.levels = levels

        # Buffers
        self.register_buffer("resolution", resolution)  # [3]
        self.register_buffer("aabbs", aabbs)  # [n_aabbs, 6]
        self.register_buffer(
            "occs", torch.zeros(self.cells_per_lvl)
        )
        self.register_buffer(
            "binaries",
            torch.zeros([levels] + resolution.tolist(), dtype=torch.bool),
        )
        
        self.pooling()
        
        
        # Grid coords & indices
        grid_coords = _meshgrid3d(resolution).reshape(
            self.cells_per_lvl, self.DIM
        )
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        grid_indices = torch.arange(self.cells_per_lvl)
        self.register_buffer("grid_indices", grid_indices, persistent=False)
    def pooling(self,):
        
        # the binaries now stored the falttened occupancy grid mipmap
        occ = self.binaries[0].float()[None,None]
        for i in range(1,self.levels):
            occ = NF.max_pool3d(occ,2,2)
            H = 2**i
            self.binaries[i] = occ[0,0].bool().repeat_interleave(H,dim=0)\
            .repeat_interleave(H,dim=1).repeat_interleave(H,dim=2)
    
    @torch.no_grad()
    def sampling(
        self,
        # rays
        rays_o: Tensor,  # [n_rays, 3]
        rays_d: Tensor,  # [n_rays, 3]
        # sigma/alpha function for skipping invisible space
        sigma_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        t_min: Optional[Tensor] = None,  # [n_rays]
        t_max: Optional[Tensor] = None,  # [n_rays]
        # rendering options
        render_step_size: float = 1e-3,
        early_stop_eps: float = 1e-4,
        alpha_thre: float = 0.0,
        stratified: bool = False,
        cone_angle: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If profided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If profided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """

        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        intervals, samples, _ = traverse_grids(
            rays_o,
            rays_d,
            self.binaries,
            self.aabbs,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices
        packed_info = samples.packed_info

        # skip invisible space
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
            sigma_fn is not None or alpha_fn is not None
        ):
            alpha_thre = min(alpha_thre, self.occs.mean().item())

            # Compute visibility of the samples, and filter out invisible samples
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert (
                    sigmas.shape == t_starts.shape
                ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
                masks = render_visibility_from_density(
                    t_starts=t_starts,
                    t_ends=t_ends,
                    sigmas=sigmas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert (
                    alphas.shape == t_starts.shape
                ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
                masks = render_visibility_from_alpha(
                    alphas=alphas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends
    
    @torch.no_grad()
    def update_every_n_steps(
        self,
        step: int,
        occs: Tensor,
        binaries: Tensor,
        n: int = 16,
    ) -> None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occs, binaries: the occupancy values and flattened binaries mipmap from the standard occupancy grid estimator
            n: how many step to update
        """
        if not self.training:
            raise RuntimeError(
                "You should only call this function only during training. "
                "Please call _update() directly if you want to update the "
                "field during inference."
            )
        if step % n == 0 and self.training:
            self.occs = occs
            self.binaries[0] = binaries
            self.pooling()