// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/extension.h>


std::tuple<RaySegmentsSpec, RaySegmentsSpec, torch::Tensor> traverse_grids(
    // rays
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor rays_mask,   // [n_rays]
    // grids
    const torch::Tensor binaries,  // [n_grids, resx, resy, resz]
    const torch::Tensor aabbs,     // [n_grids, 6]
    // intersections
    const torch::Tensor t_sorted,  // [n_rays, n_grids * 2]
    const torch::Tensor t_indices,  // [n_rays, n_grids * 2]
    const torch::Tensor hits,    // [n_rays, n_grids]
    // options
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size,
    const torch::Tensor cone_angle,
    const bool compute_intervals,
    const bool compute_samples,
    const bool compute_terminate_planes,
    const int32_t traverse_steps_limit, // <= 0 means no limit
    const bool over_allocate); // over allocate the memory for intervals and samples



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("traverse_grids", &traverse_grids);
}