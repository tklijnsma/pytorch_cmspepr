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
    torch::Tensor cond_point_count,
    torch::Tensor row_splits
    ){
    torch::NoGradGuard no_grad;

    torch::Tensor x = model_output.slice(/*dim=*/1, /*start=*/1);

    const size_t n_nodes = q.size(0);
    const auto n_dim_cluster_space = x.size(1);
    const size_t n_events = row_splits.size(0) - 1;
    float n_events_f = static_cast<float>(n_events);

    torch::Tensor grad_input = torch::zeros_like(model_output);

    for (int i_event = 0; i_event < n_events; i_event++) {
        int left = row_splits[i_event].item<int>();
        int right = row_splits[i_event + 1].item<int>();;
        float n = right - left;
        auto y_this_event = y.slice(0, left, right);

        // Number of condensation points in event (needed for L_srp)
        float n_cond = static_cast<float>(y_this_event.max().item().toFloat());

        // Calculate the indices of the condensation points in this event.
        // cond_point_count is only non-zero for condensation points, use that fact 
        auto cond_point_indices = (cond_point_count.slice(0, left, right) > 0).nonzero().squeeze(1);
        cond_point_indices += left; // Previous line has indices w.r.t. 0; add left to get indices w.r.t. left
        auto n_noise = (y_this_event == 0).sum().item().toFloat();

        for (int i = left; i < right; i++) {
            int j = which_cond_point[i].item<int>();
            
            // No attraction for noise or condensation point
            if (!((j==-1) || (j==i))){
                auto d_sq = torch::sum(torch::pow((x[i] - x[j]), 2));
                auto d = torch::sqrt(d_sq);
                auto H = huber(d + 0.00001, 4.0);

                // Attraction loss
                grad_input[i][0] += H * q[j] * d_q(beta[i]) * d_sigmoid(model_output[i][0]) / n / n_events_f;
                grad_input[j][0] += H * q[i] * d_q(beta[j]) * d_sigmoid(model_output[j][0]) / n / n_events_f;
                grad_input[i].slice(/*dim=*/0, /*start=*/1) += q[i] * q[j] * d_huber(d, 4.0) * (1.0 / (2.0 * d)) * 2.0 * (x[i] - x[j]) / n  / n_events_f;
                grad_input[j].slice(/*dim=*/0, /*start=*/1) += q[i] * q[j] * d_huber(d, 4.0) * (1.0 / (2.0 * d)) * 2.0 * (x[j] - x[i]) / n  / n_events_f;

                // Short-range potential attraction
                grad_input[i].slice(/*dim=*/0, /*start=*/1) += (
                    -beta[j] / (n_cond * cond_point_count[j])
                    * -1./torch::pow(20.0 * d_sq + 1.0, 2)
                    * 40.*(x[i] - x[j])
                    / n_events_f
                );
                grad_input[j].slice(/*dim=*/0, /*start=*/1) += (
                    -beta[j] / (n_cond * cond_point_count[j])
                    * -1./torch::pow(20.0 * d_sq + 1.0, 2)
                    * 40.*(x[j] - x[i])
                    / n_events_f
                );
                grad_input[j][0] += (
                    -1./(20.0 * d_sq + 1.0)
                    / (n_cond * cond_point_count[j])
                    * d_sigmoid(model_output[j][0])
                    / n_events_f
                );
            }

            // Repulsion loss: Loop over _other_ condensation points
            for (int k_idx = 0; k_idx < cond_point_indices.numel(); k_idx++){
                int k = cond_point_indices[k_idx].item<int>();
                if (k==j) continue; // Don't repulse from own cond point
                auto d_sq = torch::sum(torch::pow((x[i] - x[k]), 2));
                grad_input[i][0] += torch::exp(-4. * d_sq) * q[k] * d_q(beta[i]) * d_sigmoid(model_output[i][0]) / n / n_events_f;
                grad_input[k][0] += torch::exp(-4. * d_sq) * q[i] * d_q(beta[k]) * d_sigmoid(model_output[k][0]) / n / n_events_f;
                grad_input[i].slice(/*dim=*/0, /*start=*/1) += torch::exp(-4. * d_sq) * -8. * (x[i] - x[k]) * q[i] * q[k] / n / n_events_f;
                grad_input[k].slice(/*dim=*/0, /*start=*/1) += torch::exp(-4. * d_sq) * -8. * (x[k] - x[i]) * q[i] * q[k] / n / n_events_f;
            }

            // Beta noise loss
            if (y[i].item<int>() == 0){
                grad_input[i][0] += d_sigmoid(model_output[i][0]) / n_noise / n_events_f;
            }
        }

        // Beta condensation point loss
        for (int k_idx = 0; k_idx < cond_point_indices.numel(); k_idx++){
            int k = cond_point_indices[k_idx].item<int>();
            grad_input[k][0] += 
                -.2 / n_cond
                * 1./(beta[k] + 1e-9)
                * d_sigmoid(model_output[k][0])
                / n_events_f
                ;
        }

    }
    return grad_input;
}

TORCH_LIBRARY(oc_grad_cpu, m) {
  m.def("oc_grad_cpu", oc_grad_cpu);
}