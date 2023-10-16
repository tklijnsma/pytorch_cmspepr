#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <iostream>
#include <iomanip>


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define I2D(i,j,Nj) j + Nj*i


/*
Returns the squared distance between two nodes in clustering space.
*/
template <typename scalar_t>
__device__ scalar_t calc_dist_sq(
    const size_t i, // index of node i
    const size_t j, // index of node j
    const scalar_t *x, // node feature matrix
    const size_t ndim // number of dimensions 
    ){
    scalar_t distsq = 0.;
    if (i == j) return 0.;
    for (size_t idim = 0; idim < ndim; idim++) {
        scalar_t dist = x[I2D(i,idim,ndim)] - x[I2D(j,idim,ndim)];
        distsq += dist * dist;
        }
    return distsq;
    }


template <typename scalar_t> 
__global__ void oc_kernel(
    const size_t   i_event,
    // Global event info
    const scalar_t* beta, // beta per node
    const scalar_t* q,    // charge per node
    const scalar_t* x,    // cluster space coordinates
    const size_t   n_dim_cluster_space,
    const int32_t* y,
    const int32_t* which_cond_point,
    const int32_t* row_splits,
    const int32_t* cond_indices_row_splits,
    const int32_t* cond_indices,
    const int32_t* cond_counts,
    // Outputs:
    scalar_t* V_att,
    scalar_t* V_rep,
    scalar_t* V_srp
    ){
    const size_t node_start = row_splits[i_event];
    const size_t node_end = row_splits[i_event+1];
    const int32_t i_node = blockIdx.x * blockDim.x + threadIdx.x + node_start;
    if (i_node >= node_end || i_node < node_start ) return; // Safety
    const size_t n_nodes = node_end - node_start;
    const size_t cond_start = cond_indices_row_splits[i_event];
    const size_t cond_end = cond_indices_row_splits[i_event+1];
    const int32_t i_cond = which_cond_point[i_node];

    const scalar_t q_node = q[i_node];
    // For noise hits, set q_cond and d(_sq/_huber) to 0.
    const scalar_t q_cond = (i_cond == -1) ? 0. : q[i_cond];
    const scalar_t d_sq = (i_cond == -1) ? 0. : calc_dist_sq(i_node, i_cond, x, n_dim_cluster_space);
    const scalar_t d = sqrt(d_sq);
    const scalar_t d_huber = d+0.00001 <= 4.0 ?  d_sq  :  2.0 * 4.0 * (d - 4.0) ;

    const bool is_noise = (i_cond == -1);
    const bool is_cond_point = (i_node == i_cond);

    // We also need the index of the condensation point in the cond_counts array,
    // which is n_cond long.
    // This is a little confusing.
    // The condensation point has a *node index*, but it is also the nth condensation
    // point, which is the index you need to access cond_counts[n].
    int32_t nth_cond_point_index = -1;
    for (int32_t i=cond_start; i<cond_end; i++) {
        // Basically: if (cond_indices[i] == i_cond) nth_cond_point_index = i;
        nth_cond_point_index += (cond_indices[i] == i_cond) * (i+1);
        }

    // Save the count of the cluster (i.e. number of nodes in the cluster i_node belongs to)
    // Basically: is_noise ? cond_counts[nth_cond_point_index] : 1
    int32_t count_this_cond_point = 1 + (!is_noise) * (cond_counts[nth_cond_point_index]-1);

    // V_att and V_srp
    // Both will be zero for noise or if this node is a condensation point itself.
    // This is ensured by setting d_uber/d_sq/q_cond to 0. for these points.
    V_att[i_node] = d_huber * q_node * q_cond / (scalar_t)n_nodes;
    V_srp[i_node] = 1. / (20.*d_sq + 1.) / (scalar_t)(cond_end-cond_start) / (scalar_t)count_this_cond_point;

    // V_rep
    scalar_t V_rep_this = 0.;
    for (int32_t i=cond_start; i<cond_end; i++) {
        if (i == nth_cond_point_index) continue; // Don't repulse off of cond point of same cluster
        int32_t i_cond_other = cond_indices[i];
        scalar_t d_sq_other = calc_dist_sq(i_node, i_cond_other, x, n_dim_cluster_space);
        scalar_t tmp = exp(-4.0 * d_sq_other) * q_node * q[i_cond_other];
        V_rep_this += (tmp > 0.) ? tmp : 0.;
        }
    V_rep[i_node] = V_rep_this / (scalar_t)n_nodes;
}


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
    )
{
    CHECK_CUDA(beta_tensor);
    CHECK_CUDA(q_tensor);
    CHECK_CUDA(x_tensor);
    CHECK_CUDA(y_tensor);
    CHECK_CUDA(which_cond_point_tensor);
    CHECK_CUDA(row_splits_tensor);
    CHECK_CUDA(cond_indices_row_splits_tensor);
    CHECK_CUDA(cond_indices_tensor);
    CHECK_CUDA(cond_counts_tensor);

    const auto n_nodes = q_tensor.size(0);
    const auto n_dim_cluster_space = x_tensor.size(1);
    const auto n_events = row_splits_tensor.size(0) - 1;

    // Copy the event row splits over to the CPU in order to do the for-loop below
    // (cannot access row splits on CPU otherwise)
    std::vector<int32_t> cpu_row_splits(n_events+1);
    cudaMemcpy(
        &cpu_row_splits.at(0),
        row_splits_tensor.data_ptr<int32_t>(),
        (n_events+1) * sizeof(int32_t),
        cudaMemcpyDeviceToHost
        );

    // _________________________________________________________________
    // Prepare output tensor

    auto options = (
        torch::TensorOptions()
        .dtype(beta_tensor.dtype())
        .device(beta_tensor.device())
        );
    auto losses_tensor = torch::zeros({ 5 }, options);
    auto V_att_tensor = torch::zeros({n_nodes}, options);
    auto V_rep_tensor = torch::zeros({n_nodes}, options);
    auto V_srp_tensor = torch::zeros({n_nodes}, options);

    // Loop over events, parallelize over nodes
    // size_t block_size = 1024;
    size_t block_size = 24;
    size_t n_blocks;
    for (size_t i_event=0; i_event<n_events; i_event++) {
        // Nr of blocks needed to cover all nodes in event ( +blocksize-1 to round up)
        size_t n_nodes_this_event = cpu_row_splits[i_event+1] - cpu_row_splits[i_event];
        n_blocks = (n_nodes_this_event + block_size - 1) / block_size;

        AT_DISPATCH_FLOATING_TYPES(beta_tensor.type(), "oc_kernel", ([&] {
            oc_kernel <scalar_t> <<<n_blocks, block_size>>> (
                i_event,
                // Global event info
                beta_tensor.data_ptr<scalar_t>(),
                q_tensor.data_ptr<scalar_t>(),
                x_tensor.data_ptr<scalar_t>(),
                n_dim_cluster_space,
                y_tensor.data_ptr<int32_t>(),
                which_cond_point_tensor.data_ptr<int32_t>(),
                row_splits_tensor.data_ptr<int32_t>(),
                cond_indices_row_splits_tensor.data_ptr<int32_t>(),
                cond_indices_tensor.data_ptr<int32_t>(),
                cond_counts_tensor.data_ptr<int32_t>(),
                // Output
                V_att_tensor.data_ptr<scalar_t>(),
                V_rep_tensor.data_ptr<scalar_t>(),
                V_srp_tensor.data_ptr<scalar_t>()
                );
            }));
        }

    cudaDeviceSynchronize();

    losses_tensor[0] = V_att_tensor.sum() / (float_t)n_events;
    losses_tensor[1] = V_rep_tensor.sum() / (float_t)n_events;
    losses_tensor[2] = V_srp_tensor.sum() / (float_t)n_events;

    return losses_tensor;
}
