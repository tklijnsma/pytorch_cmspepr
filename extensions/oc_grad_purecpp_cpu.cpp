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
    const int i, // index of node i
    const int j, // index of node j
    const float *x, // cluster space coordinates
    const int ndim // number of dimensions 
    ){
    float distsq = 0;
    if (i == j) return 0;
    for (int d=0; d<ndim; d++) {
        float dist = x[I2D(i,d,ndim)] - x[I2D(j,d,ndim)];
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
float d_q(float beta) {
    double a = 0.0;
    double b = 1 - 1e-4;
    double c = 1.002;
    if ((beta <= a) || (beta >= b)) return 0.;
    return (
        2. / c * atanh(beta / c)
        * 1. / (1.-pow(beta / c, 2))
    );
}

/**
 * Calculate the derivative of the sigmoid function.
 *
 * @param x The input.
 * @return The derivative of the sigmoid function.
 */
float d_sigmoid(float x) {
    return exp(-x) / pow(1 + exp(-x), 2);
}


/**
 * Calculate the Huberized distance.
 *
 * @param x The input.
 * @param delta The threshold value for the Huber function.
 * @return The Huberized distance.
 */
float huber(float x, float delta) {
    if (fabs(x) < delta) {
        return pow(x, 2);
    }
    else {
        return 2 * delta * (fabs(x) - delta);
    }
}

/**
 * Calculate the derivative of the Huberized distance between two points.
 *
 * @param x The input.
 * @param delta The threshold value for the Huber function.
 * @return The derivative of the Huberized distance.
 */
float d_huber(float x, float delta) {
    if (fabs(x) < delta) {
        return 2. * x;
    }
    else {
        return 2. * delta * (x<0 ? -1 : 1);
    }
}


void calc_grad(
    float* model_output,
    float* x,
    float* beta,
    float* q,
    int* y,
    int* which_cond_point,
    int* cond_point_count,
    int* row_splits,
    // 
    int n_events,
    int n_nodes,
    int n_dim_cluster_space,
    // 
    float* grad
){
    int n_cols = n_dim_cluster_space + 1;
    float n_events_f = static_cast<float>(n_events);

    for (int i_event = 0; i_event < n_events; i_event++) {
        int left = row_splits[i_event];
        int right = row_splits[i_event + 1];
        std::cout << "Event " << i_event << " left=" << left << " right=" << right << std::endl;
        float n = right - left;

        // Count the number of condensation points in this event
        int n_cond = 0;
        std::vector<int> cond_point_indices;
        for (int i = left; i < right; i++) {
            if (cond_point_count[i] > 0) {
                cond_point_indices.push_back(i);
                n_cond++;
            }
        }
        float n_cond_f = static_cast<float>(n_cond);

        // Count the number of noise hits in this event
        int n_noise = 0;
        for (int i = left; i < right; i++) {
            if (y[i] == 0) {
                n_noise++;
            }
        }
        float n_noise_f = static_cast<float>(n_noise);

        // Loop over nodes
        for (int i = left; i < right; i++) {
            int j = which_cond_point[i];

            // No attraction for noise or condensation point
            if (!((j==-1) || (j==i))){
                float dist_sq = calc_dist_sq(i, j, x, n_dim_cluster_space);
                float dist = std::sqrt(dist_sq);
                float H = huber(dist + 0.00001, 4.0);

                // Attraction loss
                grad[I2D(i, 0, n_cols)] += H * q[j] * d_q(beta[i]) * d_sigmoid(model_output[I2D(i, 0, n_cols)]) / n / n_events_f;
                grad[I2D(j, 0, n_cols)] += H * q[i] * d_q(beta[j]) * d_sigmoid(model_output[I2D(j, 0, n_cols)]) / n / n_events_f;
                for (int d = 1; d < n_cols; d++) {
                    grad[I2D(i, d, n_cols)] += 
                        q[i] * q[j] * d_huber(dist, 4.0)
                        * (1.0 / (2.0 * dist)) * 2.0
                        * (model_output[I2D(i, d, n_cols)] - model_output[I2D(j, d, n_cols)])
                        / n / n_events_f;
                    grad[I2D(j, d, n_cols)] +=
                        q[i] * q[j] * d_huber(dist, 4.0)
                        * (1.0 / (2.0 * dist)) * 2.0
                        * (model_output[I2D(j, d, n_cols)] - model_output[I2D(i, d, n_cols)])
                        / n / n_events_f;
                }

                // Short-range potential attraction
                grad[I2D(j, 0, n_cols)] +=
                    -1./(20.0 * dist_sq + 1.0)
                    / (n_cond_f * cond_point_count[j])
                    * d_sigmoid(model_output[I2D(j, 0, n_cols)])
                    / n_events_f;
                for (int d = 1; d < n_cols; d++) {
                    float xi = model_output[I2D(i, d, n_cols)];
                    float xj = model_output[I2D(j, d, n_cols)];
                    grad[I2D(i, d, n_cols)] += 
                        -beta[j] / (n_cond_f * cond_point_count[j])
                        * -1./pow(20.0 * dist_sq + 1.0, 2)
                        * 40.*(xi - xj)
                        / n_events_f;
                    grad[I2D(j, d, n_cols)] += 
                        -beta[j] / (n_cond_f * cond_point_count[j])
                        * -1./pow(20.0 * dist_sq + 1.0, 2)
                        * 40.*(xj - xi)
                        / n_events_f;
                }
            }

            // Repulsion loss
            for(const int k : cond_point_indices){
                if (k==j) continue; // Don't repulse from own cond point
                float d_sq = calc_dist_sq(i, k, x, n_dim_cluster_space);
                grad[I2D(i, 0, n_cols)] +=
                    exp(-4. * d_sq)
                    * q[k] * d_q(beta[i])
                    * d_sigmoid(model_output[I2D(i, 0, n_cols)])
                    / n / n_events_f;
                grad[I2D(k, 0, n_cols)] +=
                    exp(-4. * d_sq)
                    * q[i] * d_q(beta[k])
                    * d_sigmoid(model_output[I2D(k, 0, n_cols)])
                    / n / n_events_f;
                for (int d = 1; d < n_cols; d++) {
                    float xi = model_output[I2D(i, d, n_cols)];
                    float xk = model_output[I2D(k, d, n_cols)];
                    grad[I2D(i, d, n_cols)] +=
                        exp(-4. * d_sq) * -8. * (xi - xk)
                        * q[i] * q[k]
                        / n / n_events_f;
                    grad[I2D(k, d, n_cols)] +=
                        exp(-4. * d_sq) * -8. * (xk - xi)
                        * q[i] * q[k]
                        / n / n_events_f;
                }
            }

            // Beta noise loss
            if (y[i] == 0){
                grad[I2D(i, 0, n_cols)] += d_sigmoid(model_output[I2D(i, 0, n_cols)]) / n_noise_f / n_events_f;
            }
        }

        // Beta condensation point loss
        for(const int k : cond_point_indices){
            grad[I2D(k, 0, n_cols)] += 
                -.2 / n_cond_f
                * 1./(beta[k] + 1e-9)
                * d_sigmoid(model_output[I2D(k, 0, n_cols)])
                / n_events_f;
        }
    }
}


torch::Tensor
oc_grad_purecpp_cpu(
    torch::Tensor model_output,
    torch::Tensor beta,
    torch::Tensor q,
    torch::Tensor y,
    torch::Tensor which_cond_point,
    torch::Tensor cond_point_count,
    torch::Tensor row_splits
    ){
    torch::NoGradGuard no_grad;
    torch::Tensor x = model_output.slice(/*dim=*/1, /*start=*/1).contiguous();

    // Output
    auto grad = torch::zeros_like(model_output);

    int n_events = row_splits.size(0) - 1;
    int n_nodes = x.size(0);
    int n_dim_cluster_space = x.size(1);

    calc_grad(
        model_output.data_ptr<float>(),
        x.data_ptr<float>(),
        beta.data_ptr<float>(),
        q.data_ptr<float>(),
        y.data_ptr<int>(),
        which_cond_point.data_ptr<int>(),
        cond_point_count.data_ptr<int>(),
        row_splits.data_ptr<int>(),
        // 
        n_events,
        n_nodes,
        n_dim_cluster_space,
        // 
        grad.data_ptr<float>()
        );

    return grad;
    }


TORCH_LIBRARY(oc_grad_purecpp_cpu, m) {
  m.def("oc_grad_purecpp_cpu", oc_grad_purecpp_cpu);
}
