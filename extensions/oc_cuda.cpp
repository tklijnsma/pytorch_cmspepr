#include <torch/extension.h>
#include <iostream>

// CUDA forward declarations
torch::Tensor oc_cuda_fn(
    torch::Tensor beta_tensor,
    torch::Tensor q_tensor,
    torch::Tensor x_tensor,
    torch::Tensor y_tensor,
    torch::Tensor which_cond_point_tensor,
    torch::Tensor row_splits_tensor,
    torch::Tensor cond_indices_row_splits_tensor,
    torch::Tensor cond_indices_tensor,
    torch::Tensor cond_counts_tensor
    );

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor oc_cuda_interface(
    torch::Tensor beta_tensor,
    torch::Tensor q_tensor,
    torch::Tensor x_tensor,
    torch::Tensor y_tensor,
    torch::Tensor which_cond_point_tensor,
    torch::Tensor row_splits_tensor,
    torch::Tensor cond_indices_row_splits_tensor,
    torch::Tensor cond_indices_tensor,
    torch::Tensor cond_counts_tensor
    ){
  CHECK_INPUT(beta_tensor);
  CHECK_INPUT(q_tensor);
  CHECK_INPUT(x_tensor);
  CHECK_INPUT(y_tensor);
  CHECK_INPUT(row_splits_tensor);

  return oc_cuda_fn(
    beta_tensor,
    q_tensor,
    x_tensor,
    y_tensor,
    which_cond_point_tensor,
    row_splits_tensor,
    cond_indices_row_splits_tensor,
    cond_indices_tensor,
    cond_counts_tensor
    );
}

TORCH_LIBRARY(oc_cuda, m) {
  m.def("oc_cuda", oc_cuda_interface);
}