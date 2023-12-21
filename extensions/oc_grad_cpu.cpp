#include <torch/extension.h>

// #include <string> //size_t, just for helper function
#include <cmath>
// #include <iostream>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) j + Nj*i

/*
Returns the squared distance between two nodes in clustering space.
*/
float calc_dist_sq(
    const size_t i, // index of node i
    const size_t j, // index of node j
    const float_t *x, // node feature matrix
    const size_t ndim // number of dimensions 
    ){
    float_t distsq = 0;
    if (i == j) return 0;
    for (size_t idim = 0; idim < ndim; idim++) {
        float_t dist = x[I2D(i,idim,ndim)] - x[I2D(j,idim,ndim)];
        distsq += dist * dist;
    }
    return distsq;
    }

/**
 * Calculate the derivative of q.
 *
 * @param beta The beta value.
 * @return The derivative of q, dq/dbeta(beta).
 */
torch::Tensor d_q(torch::Tensor beta) {
    double a = 0.0;
    double b = 1 - 1e-4;
    double c = 1.002;
    if ((beta <= a).item<bool>() || (beta >= b).item<bool>()) return torch::tensor(0.);
    return (
        2. / c * (torch::clamp(beta, a, b) / c).atanh()
        * 1./(1.-(torch::clamp(beta, a, b) / c).pow(2))
    );
}

/**
 * Calculate the derivative of the sigmoid function.
 *
 * @param x The input tensor.
 * @return The derivative of the sigmoid function.
 */
torch::Tensor d_sigmoid(torch::Tensor x) {
    return torch::exp(-x) / torch::pow(1 + torch::exp(-x), 2);
}

/**
 * Calculate the Huberized distance.
 *
 * @param x The input tensor.
 * @param delta The threshold value for the Huber function.
 * @return The Huberized distance.
 */
torch::Tensor huber(torch::Tensor x, double delta) {
    return torch::where(
        torch::abs(x) < delta,
        torch::pow(x, 2),
        2 * delta * (torch::abs(x) - delta)
    );
}

/**
 * Calculate the derivative of the Huberized distance between two points.
 *
 * @param x The input tensor.
 * @param delta The threshold value for the Huber function.
 * @return The derivative of the Huberized distance.
 */
torch::Tensor d_huber(torch::Tensor x, double delta) {
    return torch::where(
        torch::abs(x) < delta,
        2. * x,
        2. * delta * torch::sign(x)
    );
}

torch::Tensor
oc_grad_cpu(
    torch::Tensor model_output,
    torch::Tensor beta,
    torch::Tensor q,
    torch::Tensor y,
    torch::Tensor which_cond_point,
    torch::Tensor row_splits
    ){
    torch::NoGradGuard no_grad;

    torch::Tensor x = model_output.slice(/*dim=*/1, /*start=*/1);

    const size_t n_nodes = q.size(0);
    const auto n_dim_cluster_space = x.size(1);
    const size_t n_events = row_splits.size(0) - 1;

    torch::Tensor grad_input = torch::zeros_like(model_output);

    for (int i_event = 0; i_event < n_events; i_event++) {
        int left = row_splits[i_event].item<int>();
        int right = row_splits[i_event + 1].item<int>();;
        float n = right - left;
        for (int i = left; i < right; i++) {
            int j = which_cond_point[i].item<int>();
            if ((j==-1) || (j==i)) continue; // No attraction for noise or condensation point
            auto d = torch::sqrt(torch::sum(torch::pow((x[i] - x[j]), 2)));
            auto H = huber(d + 0.00001, 4.0);
            grad_input[i][0] += H * q[j] * d_q(beta[i]) * d_sigmoid(model_output[i][0]) / n;
            grad_input[j][0] += H * q[i] * d_q(beta[j]) * d_sigmoid(model_output[j][0]) / n;
            grad_input[i].slice(/*dim=*/0, /*start=*/1) += q[i] * q[j] * d_huber(d, 4.0) * (1.0 / (2.0 * d)) * 2.0 * (x[i] - x[j]) / n;
            grad_input[j].slice(/*dim=*/0, /*start=*/1) += q[i] * q[j] * d_huber(d, 4.0) * (1.0 / (2.0 * d)) * 2.0 * (x[j] - x[i]) / n;
        }
    }
    return grad_input;
}

TORCH_LIBRARY(oc_grad_cpu, m) {
  m.def("oc_grad_cpu", oc_grad_cpu);
}