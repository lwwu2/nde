#include <torch/extension.h>


void specular_bound(
    const int resolution,
    const float cutoff,
    torch::Tensor ret //[6x4xHxH]
);

void specular_filtering(
    torch::Tensor img,
    torch::Tensor bound,
    const float roughness,
    torch::Tensor ret,
    torch::Tensor weight
);

void specular_filtering_bwd(
    torch::Tensor grad,
    torch::Tensor bound,
    torch::Tensor weight,
    const float roughness,
    torch::Tensor ret
);

void fetch_mip_cube(
    torch::Tensor img,
    torch::Tensor wi,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    const int mip_level,
    int C,
    torch::Tensor ret
);

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
);

void fetch_mip_2d(
    torch::Tensor img,
    torch::Tensor xy,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor ret
);

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
);

void fetch_mip_1d(
    torch::Tensor img,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor ret
);

void fetch_mip_1d_bwd(
    torch::Tensor grad_in,
    torch::Tensor img,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor offsets,
    const int resolution,
    torch::Tensor img_grad,
    torch::Tensor x_grad,
    torch::Tensor r_grad
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("specular_bound", &specular_bound);
    m.def("specular_filtering", &specular_filtering);
    m.def("specular_filtering_bwd", &specular_filtering_bwd);
    m.def("fetch_mip_cube", &fetch_mip_cube);
    m.def("fetch_mip_cube_bwd", &fetch_mip_cube_bwd);
    m.def("fetch_mip_2d", &fetch_mip_2d);
    m.def("fetch_mip_2d_bwd", &fetch_mip_2d_bwd);
    m.def("fetch_mip_1d", &fetch_mip_1d);
    m.def("fetch_mip_1d_bwd", &fetch_mip_1d_bwd);
}