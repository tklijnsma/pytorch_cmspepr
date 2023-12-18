from typing import Tuple
import torch

from torch_cmspepr import _loaded_ops
from .utils import batch_to_row_splits


def calc_q_betaclip(beta, qmin=1.0):
    """
    Clip beta values.

    Parameters:
        beta (ndarray): An array of beta values.
        qmin (float, optional): The minimum value of q. Defaults to 1.0.

    Returns:
        ndarray: An array of clipped beta values.
    """
    return (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin


@torch.jit.script
def analyze_cond_points(
    q: torch.FloatTensor, y: torch.IntTensor, row_splits: torch.IntTensor
) -> Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor, torch.IntTensor]:
    n_events = int(row_splits.size(0)) - 1
    n_nodes = int(y.size(0))
    device = y.device

    # Count total number of cond points and nr of cond points per event
    n_cond_per_event = torch.zeros(n_events, dtype=torch.int, device=device)
    cond_row_splits = torch.zeros(n_events + 1, dtype=torch.int, device=device)
    for i_event in range(n_events):
        n_cond = y[row_splits[i_event] : row_splits[i_event + 1]].max()
        n_cond_per_event[i_event] = n_cond
        cond_row_splits[i_event + 1] = cond_row_splits[i_event] + n_cond
    n_cond_total = n_cond_per_event.sum()

    cond_indices = torch.zeros(n_cond_total, dtype=torch.int, device=device)
    cond_counts = torch.zeros(n_cond_total, dtype=torch.int, device=device)
    which_cond_point = -1 * torch.ones(n_nodes, dtype=torch.int, device=device)

    for i_event in range(n_events):
        left = row_splits[i_event]
        right = row_splits[i_event + 1]
        n_cond = n_cond_per_event[i_event]
        i_max = torch.zeros(n_cond + 1, device=device, dtype=torch.int)
        i_max[0] = -1  # Let noise hits point to index -1 (meaning "no cond point")
        q_max = torch.zeros(n_cond + 1, device=device, dtype=torch.float)
        counts = torch.zeros(n_cond + 1, device=device, dtype=torch.int)

        for i_node in range(int(left), int(right)):
            if y[i_node] == 0:
                continue  # Noise has no cond point
            counts[y[i_node]] += 1
            if q_max[y[i_node]] < q[i_node]:
                q_max[y[i_node]] = float(q[i_node])
                i_max[y[i_node]] = i_node

        # Fill in to which cond point the nodes belong
        which_cond_point[left:right] = i_max[y[left:right]]
        # (i_max/counts[0] is noise, don't use it)
        cond_indices[cond_row_splits[i_event] : cond_row_splits[i_event + 1]] = i_max[
            1:
        ]
        cond_counts[cond_row_splits[i_event] : cond_row_splits[i_event + 1]] = counts[
            1:
        ]

    return cond_indices, cond_counts, cond_row_splits, which_cond_point


@torch.jit.script
def cond_point_indices_and_counts(
    q: torch.FloatTensor, y: torch.IntTensor, row_splits: torch.IntTensor
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    n_events = int(row_splits.size(0)) - 1
    n_nodes = int(y.size(0))
    device = y.device

    cond_point_index = -1 * torch.ones(n_nodes, dtype=torch.int, device=device)
    cond_point_count = torch.zeros(n_nodes, dtype=torch.int, device=device)

    for i_event in range(n_events):
        left = row_splits[i_event]
        right = row_splits[i_event + 1]

        n_cond = y[left:right].max()

        # Per cond point, open arrays that contain:
        # - the node index of the cond point
        # - the q value of the cond point
        # - the number of nodes belonging to the cond point
        i_max = torch.zeros(n_cond, device=device, dtype=torch.int)
        q_max = torch.zeros(n_cond, device=device, dtype=torch.float)
        counts = torch.zeros(n_cond, device=device, dtype=torch.int)

        for i_node in range(int(left), int(right)):
            y_ = y[i_node]  # Truth cluster number of the node
            if y[i_node] == 0:
                continue  # Noise has no cond point, continue
            # Subtract 1: Put cluster 1 in index 0, 2 in 1, etc.
            counts[y_ - 1] += 1
            if q_max[y_ - 1] < q[i_node]:
                q_max[y_ - 1] = float(q[i_node])
                i_max[y_ - 1] = i_node

        # Fill in to which cond point the nodes belong
        signal_node_indices = left + torch.nonzero(y[left:right]).squeeze(dim=-1)
        cond_point_index[signal_node_indices] = i_max[y[signal_node_indices] - 1]

        # Fill in the number of nodes per cond point
        # cond_point_count will be zero for any non-cond point
        for i in range(int(n_cond)):
            cond_point_count[i_max[i]] = counts[i]

    return cond_point_index, cond_point_count


# JIT compile the interface to the extensions.
# Do not try to compile torch.ops.oc_* if those ops aren't actually loaded!
if 'oc_cuda.so' in _loaded_ops:

    @torch.jit.script
    def oc_cuda(
        beta,
        q,
        x,
        y,
        which_cond_point,
        row_splits,
        cond_row_splits,
        cond_indices,
        cond_counts,
    ) -> torch.Tensor:
        return torch.ops.oc_cuda.oc_cuda(
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

else:

    @torch.jit.script
    def oc_cuda(
        beta,
        q,
        x,
        y,
        which_cond_point,
        row_splits,
        cond_row_splits,
        cond_indices,
        cond_counts,
    ) -> torch.Tensor:
        raise Exception('CUDA extension for oc not installed')


if 'oc_cpu.so' in _loaded_ops:

    @torch.jit.script
    def oc_cpu(beta, q, x, y, row_splits) -> torch.Tensor:
        return torch.ops.oc_cpu.oc_cpu(beta, q, x, y, row_splits)

else:

    @torch.jit.script
    def oc_cpu(beta, q, x, y, row_splits) -> torch.Tensor:
        raise Exception('CPU extension for oc not installed')


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
        return oc_cpu(beta, q, x, y.type(torch.int), row_splits)
    else:
        # GPU needs more prep work in python
        # Determine condensation point indices, counts, and event boundaries
        y = y.type(torch.int)
        (
            cond_indices,
            cond_counts,
            cond_row_splits,
            which_cond_point,
        ) = analyze_cond_points(q, y, row_splits)
        losses = oc_cuda(
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
        n_events = int(len(row_splits) - 1)
        L_beta_cond_logterm = torch.tensor(0.0)
        L_beta_noise = torch.tensor(0.0)
        is_noise = y == 0
        for i_event in range(n_events):
            # L_beta_cond_logterm
            # Get beta of all cond points in this event
            beta_cond = beta[
                cond_indices[cond_row_splits[i_event] : cond_row_splits[i_event + 1]]
            ]
            L_beta_cond_logterm += (-0.2 * torch.log(beta_cond + 0.000000001)).mean()
            # L_beta_noise: get all beta's for y==0 in _this_ event
            beta_noise = beta[row_splits[i_event] : row_splits[i_event + 1]][
                is_noise[row_splits[i_event] : row_splits[i_event + 1]]
            ]
            L_beta_noise += beta_noise.mean()

        losses[3] = L_beta_cond_logterm / float(n_events)
        losses[4] = L_beta_noise / float(n_events)
        return losses



def oc_noext(
    beta: torch.FloatTensor,
    q: torch.FloatTensor,
    x: torch.FloatTensor,
    y: torch.LongTensor,  # Use long for consistency
    batch: torch.LongTensor,  # Use long for consistency
) -> torch.FloatTensor:
    """
    Calculate the object condensation loss function.
    Uses no extensions.

    Args:
        beta (torch.FloatTensor): The beta values.
        q (torch.FloatTensor): The q values.
        x (torch.FloatTensor): The x values.
        y (torch.LongTensor): The y values.
        batch (torch.LongTensor): The batch values.

    Returns:
        torch.FloatTensor: The loss values: V_att, V_rep, V_srp, L_beta_cond_logterm,
            and L_beta_noise
    """
    n_events = int(batch.max() + 1)
    N = int(beta.size(0))

    # Translate batch vector into row splits
    row_splits = batch_to_row_splits(batch).type(torch.int)

    y = y.type(torch.int)
    cond_point_index, cond_point_count = cond_point_indices_and_counts(q, y, row_splits)

    is_noise = y == 0

    V_att = torch.zeros(N)
    V_srp = torch.zeros(N)
    V_rep = torch.zeros(N)

    L_beta_cond_logterm = torch.zeros(n_events)
    L_beta_noise = torch.zeros(n_events)

    for i_event in range(n_events):
        left = int(row_splits[i_event])
        right = int(row_splits[i_event + 1])

        # Number of nodes and number of condensation points in this event
        n = float(row_splits[i_event + 1] - row_splits[i_event])
        n_cond = float(y[left:right].max())

        # Indices of the condensation points in this event
        cond_point_indices = left + (
            cond_point_count[left:right] > 0
        ).nonzero().squeeze(dim=-1)
        # Indices of the noise nodes in this event
        noise_indices = left + is_noise[left:right].nonzero().squeeze(dim=-1)

        for i in range(left, right):
            i_cond = cond_point_index[i]

            # V_att and V_srp
            if i_cond == -1 or i == i_cond:
                # Noise point or condensation point: V_att and V_srp are 0
                pass
            else:
                d_sq = torch.sum((x[i] - x[i_cond]) ** 2)
                d = torch.sqrt(d_sq)
                d_plus_eps = d + 0.00001
                d_huber = (
                    d_plus_eps**2
                    if d_plus_eps <= 4.0
                    else 2.0 * 4.0 * (d_plus_eps - 4.0)
                )
                V_att[i] = d_huber * q[i] * q[i_cond] / n
                V_srp[i] = (
                    -beta[i_cond]
                    / (20.0 * d_sq + 1.0)
                    / float(
                        cond_point_count[i_cond]
                    )  # Number of nodes belonging to cond point
                    / n_cond  # Number of condensation points in event
                )

            # V_rep
            for i_cond_other in cond_point_indices:
                if i_cond_other == i_cond:
                    continue  # Don't repulse from own cond point
                d_sq = torch.sum((x[i] - x[i_cond_other]) ** 2)
                V_rep[i] += torch.exp(-4.0 * d_sq) * q[i] * q[i_cond_other] / n

        L_beta_cond_logterm[i_event] = (
            -0.2 * torch.log(beta[cond_point_indices] + 1e-9).mean()
        )
        L_beta_noise[i_event] = beta[noise_indices].mean()

    n_events = float(n_events)
    return (
        torch.stack(
            [
                V_att.sum(),
                V_rep.sum(),
                V_srp.sum(),
                L_beta_cond_logterm.sum(),
                L_beta_noise.sum(),
            ]
        )
        / n_events
    )


@torch.jit.script
def oc_noext_jit(
    beta: torch.FloatTensor,
    q: torch.FloatTensor,
    x: torch.FloatTensor,
    y: torch.LongTensor,  # Use long for consistency
    batch: torch.LongTensor,  # Use long for consistency
) -> torch.FloatTensor:
    return oc_noext(beta, q, x, y, batch)
