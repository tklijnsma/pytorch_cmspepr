import os.path as osp
from math import log
import pytest
import torch
from torch_geometric.data import Data

import torch_cmspepr

gpu = torch.device('cuda')

SO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
CPU_INSTALLED = osp.isfile(osp.join(SO_DIR, 'oc_cpu.so'))
CUDA_INSTALLED = osp.isfile(osp.join(SO_DIR, 'oc_cuda.so'))

# fmt: off
# Generate some carefully crafted test data
# Single event
class single:
    model_out = torch.FloatTensor([
        # Event 0
        # beta x0    x1        y
        [0.01, 0.40, 0.40],  # 0
        [0.02, 0.10, 0.90],  # 0
        [0.12, 0.70, 0.70],  # 1 <- d_sq to cond point = 0.02^2 + 0.02^2 = 0.0008; d=0.0283
        [0.01, 0.90, 0.10],  # 0
        [0.13, 0.72, 0.72],  # 1 <-- cond point for y=1
        ])
    x = model_out[:, 1:].contiguous()
    y = torch.LongTensor([0, 0, 1, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0])
    beta = torch.sigmoid(model_out[:, 0]).contiguous()
    q = torch_cmspepr.calc_q_betaclip(beta)

    @classmethod
    def d_sq(cls, i, j):
        return ((cls.x[i] - cls.x[j]) ** 2).sum()

    @classmethod
    def d(cls, i, j):
        return torch.sqrt(((cls.x[i] - cls.x[j]) ** 2).sum())

    @classmethod
    def losses(cls):
        # Calculates the OC loss manually
        beta = single.beta
        q = single.q
        d = single.d
        d_sq = single.d_sq

        V_att = (d(2, 4)+1e-5)**2 * q[2] * q[4] / 5.0  # Since d is small, d == d_huber
        V_rep = (
            torch.exp(-4.0 * d_sq(0, 4)) * q[0] * q[4]
            + torch.exp(-4.0 * d_sq(1, 4)) * q[1] * q[4]
            + torch.exp(-4.0 * d_sq(3, 4)) * q[3] * q[4]
        ) / 5.0
        V_srp = -1.0 / (20.0 * d_sq(2, 4) + 1.0) * beta[4] / 2.0
        L_beta_cond_logterm = -0.2 * log(beta[4] + 1e-9)
        L_beta_noise = (beta[0] + beta[1] + beta[3]) / 3.0

        losses_man = torch.FloatTensor(
            [V_att, V_rep, V_srp, L_beta_cond_logterm, L_beta_noise]
        )
        return losses_man

# Model output for two events with a batch tensor
class double:
    model_out = torch.FloatTensor([
        # Event 0
        # beta x0    x1       idx y
        [0.01, 0.40, 0.40],  #  0 0
        [0.02, 0.10, 0.90],  #  1 0
        [0.12, 0.70, 0.70],  #  2 1 <- d_sq to cond point = 0.02^2 + 0.02^2 = 0.0008; d=0.0283
        [0.01, 0.90, 0.10],  #  3 0
        [0.13, 0.72, 0.72],  #  4 1 <-- cond point for y=1
        # Event 1
        [0.11, 0.40, 0.40],  #  5 2
        [0.02, 0.10, 0.90],  #  6 0
        [0.12, 0.70, 0.70],  #  7 1 <-- cond point for y=1
        [0.01, 0.90, 0.10],  #  8 0
        [0.13, 0.72, 0.72],  #  9 2 <-- cond point for y=2
        [0.11, 0.72, 0.72],  # 10 1
        ])
    x = model_out[:,1:].contiguous()
    y = torch.LongTensor([
        0, 0, 1, 0, 1,    # Event 0
        2, 0, 1, 0, 2, 1  # Event 1
        ])
    batch = torch.LongTensor([
        0, 0, 0, 0, 0,    # Event 0
        1, 1, 1, 1, 1, 1  # Event 1
        ])
    # fmt: on
    row_splits = torch.IntTensor([0, 5, 11])
    beta = torch.sigmoid(model_out[:, 0]).contiguous()
    q = torch_cmspepr.calc_q_betaclip(beta).contiguous()

    cond_indices = torch.IntTensor([4, 7, 9])
    cond_counts = torch.IntTensor([2, 2, 2])
    cond_row_splits = torch.IntTensor([0, 1, 3])
    # fmt: off
    which_cond_point = torch.IntTensor([
        -1, -1, 4, -1, 4,
        9,  -1, 7, -1, 9, 7
        ])

    @classmethod
    def d_sq(cls, i, j):
        return ((cls.x[i] - cls.x[j]) ** 2).sum()

    @classmethod
    def d(cls, i, j):
        return torch.sqrt(((cls.x[i] - cls.x[j]) ** 2).sum())

    @classmethod
    def losses(cls):
        # Calculates the OC loss manually
        beta = double.beta
        q = double.q
        d = double.d
        d_sq = double.d_sq

        # Event 0
        V_att_0 = (d(2, 4)+1e-5)**2 * q[2] * q[4] / 5.0  # Since d is small, d == d_huber
        V_rep_0 = (
            torch.exp(-4.0 * d_sq(0, 4)) * q[0] * q[4]
            + torch.exp(-4.0 * d_sq(1, 4)) * q[1] * q[4]
            + torch.exp(-4.0 * d_sq(3, 4)) * q[3] * q[4]
        ) / 5.0
        V_srp_0 = -beta[4] / (20.0 * d_sq(2, 4) + 1.0) / 2.0
        L_beta_cond_logterm_0 = -0.2 * log(beta[4] + 1e-9)
        L_beta_noise_0 = (beta[0] + beta[1] + beta[3]) / 3.0

        # Event 1
        V_att_1 = (
            (d(5, 9)+1e-5)**2 * q[5] * q[9]
            + (d(10, 7)+1e-5)**2 * q[10] * q[7]
            ) / 6.0  # Since d is small, d == d_huber
        V_rep_1 = (
              torch.exp(-4.0 * d_sq(5, 7)) * q[5] * q[7]
            + torch.exp(-4.0 * d_sq(6, 7)) * q[6] * q[7]
            + torch.exp(-4.0 * d_sq(8, 7)) * q[8] * q[7]
            + torch.exp(-4.0 * d_sq(9, 7)) * q[9] * q[7]
            + torch.exp(-4.0 * d_sq(6, 9)) * q[6] * q[9]
            + torch.exp(-4.0 * d_sq(7, 9)) * q[7] * q[9]
            + torch.exp(-4.0 * d_sq(8, 9)) * q[8] * q[9]
            + torch.exp(-4.0 * d_sq(10, 9)) * q[10] * q[9]
        ) / 6.0

        V_srp_1 =  -beta[9] / (20.0 * d_sq(5, 9) + 1.0) / 2.0
        V_srp_1 += -beta[7] / (20.0 * d_sq(10, 7) + 1.0) / 2.0
        V_srp_1 /= 2. # Number of cond points in event

        L_beta_cond_logterm_1 = -0.2 * (log(beta[7] + 1e-9) + log(beta[9] + 1e-9)) / 2.
        L_beta_noise_1 = (beta[6] + beta[8]) / 2.0

        losses_man = torch.FloatTensor([
            V_att_0 + V_att_1,
            V_rep_0 + V_rep_1,
            V_srp_0 + V_srp_1,
            L_beta_cond_logterm_0 + L_beta_cond_logterm_1,
            L_beta_noise_0 + L_beta_noise_1
            ]) / 2.
        return losses_man
# fmt: on

if torch.cuda.is_available():
    class double_gpu:
        model_out = double.model_out.to(gpu)
        x = double.x.to(gpu)
        y = double.y.type(torch.int).to(gpu)
        batch = double.batch.to(gpu)
        row_splits = double.row_splits.to(gpu)
        beta = double.beta.to(gpu)
        q = double.q.to(gpu)
        cond_indices = double.cond_indices.to(gpu)
        cond_counts = double.cond_counts.to(gpu)
        cond_row_splits = double.cond_row_splits.to(gpu)
        which_cond_point = double.which_cond_point.to(gpu)


def test_oc_noext_single():
    print(f'{single.beta=}')
    print(f'{single.q=}')
    print(f'{single.x=}')
    print(f'{single.y=}')
    print(f'{single.batch=}')
    losses = torch_cmspepr.oc_noext(single.beta, single.q, single.x, single.y, single.batch)
    losses_man = single.losses()
    print(f'{losses=}')
    print(f'{losses_man=}')
    assert torch.allclose(losses, losses_man, rtol=0.001, atol=0.001)

def test_oc_noext_double():
    import torch_cmspepr
    losses = torch_cmspepr.oc_noext(double.beta, double.q, double.x, double.y, double.batch)
    losses_man = double.losses()
    print(f'{losses=}')
    print(f'{losses_man=}')
    assert torch.allclose(losses, losses_man, rtol=0.001, atol=0.001)

@pytest.mark.skipif(
    not CPU_INSTALLED,
    reason='CPU extension for oc not installed',
)
def test_oc_cpu_single():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))

    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        single.beta,
        single.q,
        single.x,
        single.y.type(torch.int),
        torch.IntTensor([0, 5]),
    )

    losses_man = single.losses()
    print(f'{losses_man=}')
    print(f'{losses_cpp=}')
    assert torch.allclose(losses_cpp, losses_man, rtol=0.001, atol=0.001)


def test_oc_interface_single():
    import torch_cmspepr

    losses = torch_cmspepr.oc(single.beta, single.q, single.x, single.y, single.batch)
    losses_man = single.losses()
    print(f'{losses_man=}')
    print(f'{losses=}')
    assert torch.allclose(losses, losses_man, rtol=0.001, atol=0.001)


def test_analyze_cond_points():
    from torch_cmspepr.objectcondensation import analyze_cond_points

    cond_indices, cond_counts, cond_row_splits, which_cond_point = analyze_cond_points(
        double.q, double.y.type(torch.int), double.row_splits
    )
    print(f'{cond_indices=}')
    print(f'{cond_counts=}')
    print(f'{cond_row_splits=}')
    print(f'{which_cond_point=}')
    assert torch.allclose(cond_indices, double.cond_indices)
    assert torch.allclose(cond_counts, double.cond_counts)
    assert torch.allclose(cond_row_splits, double.cond_row_splits)
    assert torch.allclose(which_cond_point, double.which_cond_point)


@pytest.mark.skipif(
    not CPU_INSTALLED,
    reason='CPU extension for oc not installed',
)
def test_oc_cpu_double():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))

    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        double.beta,
        double.q,
        double.x,
        double.y.type(torch.int),
        double.row_splits,
    )
    losses_man = double.losses()
    print(losses_man)
    print(losses_cpp)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses_cpp, losses_man, rtol=0.01, atol=0.01)


@pytest.mark.skipif(
    not CPU_INSTALLED,
    reason='CPU extension for oc not installed',
)
def test_oc_python_batch():
    import torch_cmspepr
    losses = torch_cmspepr.oc(
        double.beta,
        double.q,
        double.x,
        double.y.type(torch.int),
        double.batch,
    )
    losses_man = double.losses()
    print(losses)
    print(losses_man)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses_man, losses, rtol=0.01, atol=0.01)


@pytest.mark.skipif(
    not CUDA_INSTALLED,
    reason='CUDA extension for oc not installed',
)
def test_oc_gpu_batch():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cuda.so'))
    print('Running CUDA extension')
    losses_cuda = torch.ops.oc_cuda.oc_cuda(
        double_gpu.beta,
        double_gpu.q,
        double_gpu.x,
        double_gpu.y,
        double_gpu.which_cond_point,
        double_gpu.row_splits,
        double_gpu.cond_row_splits,
        double_gpu.cond_indices,
        double_gpu.cond_counts,
    ).cpu()
    print(f'{losses_cuda=}')
    losses_man = double.losses()
    print(f'{losses_man=}')
    # Don't compare L_beta_cond_logterm and L_noise losses here
    assert torch.allclose(losses_cuda[:3], losses_man[:3])


@pytest.mark.skipif(
    not CUDA_INSTALLED,
    reason='CUDA extension for oc not installed',
)
def test_oc_interface_gpu_double():
    import torch_cmspepr

    print('Running CUDA extension')
    losses_cuda = torch_cmspepr.oc(
        double_gpu.beta,
        double_gpu.q,
        double_gpu.x,
        double_gpu.y,
        double_gpu.batch,
    ).cpu()
    print(f'{losses_cuda=}')
    losses_man = double.losses()
    print(f'{losses_man=}')
    assert torch.allclose(losses_cuda, losses_man)
