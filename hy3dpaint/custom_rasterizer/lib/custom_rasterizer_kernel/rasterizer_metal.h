#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rasterize_image_metal(
    torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);
