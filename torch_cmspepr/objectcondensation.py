from typing import Tuple
import torch

from .utils import batch_to_row_splits


@torch.jit.script
def analyze_cond_points(
    q: torch.FloatTensor,
    y: torch.IntTensor,
    row_splits: torch.IntTensor
    ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor, torch.IntTensor]:
    n_events = int(row_splits.size(0)) - 1
    n_nodes = int(y.size(0))
    device = y.device

    # Count total number of cond points and nr of cond points per event
    n_cond_per_event = torch.zeros(n_events, dtype=torch.int, device=device)
    cond_row_splits = torch.zeros(n_events+1, dtype=torch.int, device=device)
    for i_event in range(n_events):
        n_cond = y[row_splits[i_event]:row_splits[i_event+1]].max()
        n_cond_per_event[i_event] = n_cond
        cond_row_splits[i_event+1] = cond_row_splits[i_event] + n_cond
    n_cond_total = n_cond_per_event.sum()

    cond_indices = torch.zeros(n_cond_total, dtype=torch.int, device=device)
    cond_counts = torch.zeros(n_cond_total, dtype=torch.int, device=device)
    which_cond_point = -1 * torch.ones(n_nodes, dtype=torch.int, device=device)

    for i_event in range(n_events):
        left = row_splits[i_event]
        right = row_splits[i_event+1]
        n_cond = n_cond_per_event[i_event]
        i_max = torch.zeros(n_cond+1, device=device, dtype=torch.int)
        i_max[0] = -1 # Let noise hits point to index -1 (meaning "no cond point")
        q_max = torch.zeros(n_cond+1, device=device, dtype=torch.float)
        counts = torch.zeros(n_cond+1, device=device, dtype=torch.int)

        for i_node in range(int(left), int(right)):
            if y[i_node] == 0: continue # Noise has no cond point
            counts[y[i_node]] += 1
            if q_max[y[i_node]] < q[i_node]:
                q_max[y[i_node]] = float(q[i_node])
                i_max[y[i_node]] = i_node

        # Fill in to which cond point the nodes belong
        which_cond_point[left:right] = i_max[y[left:right]]
        # (i_max/counts[0] is noise, don't use it)
        cond_indices[cond_row_splits[i_event]:cond_row_splits[i_event+1]] = i_max[1:]
        cond_counts[cond_row_splits[i_event]:cond_row_splits[i_event+1]] = counts[1:]

    return cond_indices, cond_counts, cond_row_splits, which_cond_point


@torch.jit.script
def oc(
    beta: torch.FloatTensor,
    q: torch.FloatTensor,
    x: torch.FloatTensor,
    y: torch.LongTensor,  # Use long for consistency
    batch: torch.LongTensor,  # Use long for consistency
    sB: float = 1.0,
):
    """
    Calculate the object condensation loss function.

    Args:
        beta (torch.FloatTensor): Beta as described in https://arxiv.org/abs/2002.03605;
            simply a sigmoid of the raw model output
        q (torch.FloatTensor): Charge q per node; usually a function of beta.
        x (torch.FloatTensor): Latent clustering space coordinates for every node.
        y (torch.LongTensor): Clustering truth. WARNING: The torch.op expects y to be
            nicely *incremental*. There should not be any holes in it.
        batch (torch.LongTensor): Batch vector to designate event boundaries. WARNING:
            It is expected that batch is *sorted*.

    Returns:
        torch.FloatTensor: A len-5 tensor with the 5 loss components of the OC loss
            function: V_att, V_rep, V_srp, L_beta_cond_logterm, and L_beta_noise. The
            full OC loss is simply the sum of this tensor.
    """
    N = beta.size(0)
    assert beta.dim() == 1
    assert q.dim() == 1
    assert beta.size() == q.size()
    assert x.size(0) == N
    assert y.size(0) == N
    assert batch.size(0) == N
    device = beta.device

    # Translate batch vector into row splits
    row_splits = batch_to_row_splits(batch).type(torch.int)

    if device == torch.device('cpu'):
        return torch.ops.oc_cpu.oc_cpu(beta, q, x, y.type(torch.int), row_splits)
    else:
        # GPU needs more prep work in python
        # Determine condensation point indices, counts, and event boundaries
        y = y.type(torch.int)
        cond_indices, cond_counts, cond_row_splits, which_cond_point = analyze_cond_points(q, y, row_splits)
        losses = torch.ops.oc_cuda.oc_cuda(
            beta,
            q,
            x,
            y,
            which_cond_point,
            row_splits,
            cond_row_splits,
            cond_indices,
            cond_counts,
            )
        # L_beta_cond_logterm and L_beta_noise are not calculated in CUDA extension
        n_events = int(len(row_splits)-1)
        L_beta_cond_logterm = 0.
        L_beta_noise = 0.
        is_noise = y==0
        for i_event in range(n_events):
            # L_beta_cond_logterm
            # Get beta of all cond points in this event
            beta_cond = beta[cond_indices[cond_row_splits[i_event]:cond_row_splits[i_event+1]]]
            L_beta_cond_logterm += (-0.2 * torch.log(beta_cond + 0.000000001)).mean()
            # L_beta_noise: get all beta's for y==0 in _this_ event
            beta_noise = beta[row_splits[i_event]:row_splits[i_event+1]][is_noise[row_splits[i_event]:row_splits[i_event+1]]]
            L_beta_noise += beta_noise.mean()

        losses[3] = L_beta_cond_logterm / float(n_events)
        losses[4] = L_beta_noise / float(n_events)
        return losses
