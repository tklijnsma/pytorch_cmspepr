import os.path as osp
from math import log

import torch
from torch_geometric.data import Data


gpu = torch.device('cuda')

SO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def calc_q_betaclip(beta, qmin=1.0):
    return (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin


def oc_cmspepr_hgcal_core(inst):
    """
    Runs the pure-Python OC implementation from cmspepr_hgcal_core.
    """
    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.0

    loss_py = objectcondensation.oc_loss(
        inst.model_out, Data(y=inst.y, batch=inst.batch)
    )
    losses_py = torch.FloatTensor(
        [
            loss_py["V_att"],
            loss_py["V_rep"],
            loss_py["L_beta_sig"],
            loss_py["L_beta_cond_logterm"],
            loss_py["L_beta_noise"],
        ]
    )
    return losses_py


# Single event
class single:
    # fmt: off
    model_out = torch.FloatTensor([
        # Event 0
        # beta x0    x1        y
        [0.01, 0.40, 0.40],  # 0
        [0.02, 0.10, 0.90],  # 0
        [0.12, 0.70, 0.70],  # 1 <- d_sq to cond point = 0.02^2 + 0.02^2 = 0.0008; d=0.0283
        [0.01, 0.90, 0.10],  # 0
        [0.13, 0.72, 0.72],  # 1 <-- cond point for y=1
        ])
    # fmt: on
    x = model_out[:, 1:].contiguous()
    y = torch.LongTensor([0, 0, 1, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0])
    beta = torch.sigmoid(model_out[:, 0]).contiguous()
    q = calc_q_betaclip(beta)

    @classmethod
    def d(cls, i, j):
        return ((cls.x[i] - cls.x[j]) ** 2).sum()

    # Manual OC:
    @classmethod
    def losses(cls):
        beta = single.beta
        q = single.q
        d = single.d

        V_att = d(2, 4) * q[2] * q[4] / 5.0  # Since d is small, d == d_huber
        V_rep = (
            torch.exp(-4.0 * d(0, 4)) * q[0] * q[4]
            + torch.exp(-4.0 * d(1, 4)) * q[1] * q[4]
            + torch.exp(-4.0 * d(3, 4)) * q[3] * q[4]
        ) / 5.0
        V_srp = -1.0 / (20.0 * d(2, 4) + 1.0) * beta[4] / 2.0
        L_beta_cond_logterm = -0.2 * log(beta[4] + 1e-9)
        L_beta_noise = (beta[0] + beta[1] + beta[3]) / 3.0

        losses_man = torch.FloatTensor(
            [V_att, V_rep, V_srp, L_beta_cond_logterm, L_beta_noise]
        )
        return losses_man


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


def test_oc_python_single():
    import torch_cmspepr

    losses = torch_cmspepr.oc(single.beta, single.q, single.x, single.y, single.batch)
    losses_man = single.losses()
    print(f'{losses_man=}')
    print(f'{losses=}')
    assert torch.allclose(losses, losses_man, rtol=0.001, atol=0.001)


class multiple:
    # fmt: off
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
    q = calc_q_betaclip(beta).contiguous()

    cond_indices = torch.IntTensor([4, 7, 9])
    cond_counts = torch.IntTensor([2, 2, 2])
    cond_row_splits = torch.IntTensor([0, 1, 3])
    # fmt: off
    which_cond_point = torch.IntTensor([
        -1, -1, 4, -1, 4,
        9,  -1, 7, -1, 9, 7
        ])
    # fmt: on


class multiple_gpu:
    model_out = multiple.model_out.to(gpu)
    x = multiple.x.to(gpu)
    y = multiple.y.type(torch.int).to(gpu)
    batch = multiple.batch.to(gpu)
    row_splits = multiple.row_splits.to(gpu)
    beta = multiple.beta.to(gpu)
    q = multiple.q.to(gpu)
    cond_indices = multiple.cond_indices.to(gpu)
    cond_counts = multiple.cond_counts.to(gpu)
    cond_row_splits = multiple.cond_row_splits.to(gpu)
    which_cond_point = multiple.which_cond_point.to(gpu)


def test_analyze_cond_points():
    from torch_cmspepr.objectcondensation import analyze_cond_points

    cond_indices, cond_counts, cond_row_splits, which_cond_point = analyze_cond_points(
        multiple.q, multiple.y.type(torch.int), multiple.row_splits
    )
    print(f'{cond_indices=}')
    print(f'{cond_counts=}')
    print(f'{cond_row_splits=}')
    print(f'{which_cond_point=}')
    assert torch.allclose(cond_indices, multiple.cond_indices)
    assert torch.allclose(cond_counts, multiple.cond_counts)
    assert torch.allclose(cond_row_splits, multiple.cond_row_splits)
    assert torch.allclose(which_cond_point, multiple.which_cond_point)


def test_oc_cpu_batch():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))
    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.0

    loss_py = objectcondensation.oc_loss(
        multiple.model_out, Data(y=multiple.y, batch=multiple.batch)
    )
    losses_py = torch.FloatTensor(
        [
            loss_py["V_att"],
            loss_py["V_rep"],
            loss_py["L_beta_sig"],
            loss_py["L_beta_cond_logterm"],
            loss_py["L_beta_noise"],
        ]
    )
    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        multiple.beta,
        multiple.q,
        multiple.x,
        multiple.y.type(torch.int),
        multiple.row_splits,
    )
    print(losses_py)
    print(losses_cpp)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses_cpp, losses_py, rtol=0.01, atol=0.01)


def test_oc_python_batch():
    import torch_cmspepr

    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.0

    loss_py = objectcondensation.oc_loss(
        multiple.model_out, Data(y=multiple.y, batch=multiple.batch)
    )
    losses_py = torch.FloatTensor(
        [
            loss_py["V_att"],
            loss_py["V_rep"],
            loss_py["L_beta_sig"],
            loss_py["L_beta_cond_logterm"],
            loss_py["L_beta_noise"],
        ]
    )
    losses = torch_cmspepr.oc(
        multiple.beta,
        multiple.q,
        multiple.x,
        multiple.y.type(torch.int),
        multiple.batch,
    )
    print(losses_py)
    print(losses)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses, losses_py, rtol=0.01, atol=0.01)


def test_oc_gpu_batch():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cuda.so'))
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))
    print('Running CPU extension')
    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        multiple.beta,
        multiple.q,
        multiple.x,
        multiple.y.type(torch.int),
        multiple.row_splits,
    )
    print('Running CUDA extension')
    losses_cuda = torch.ops.oc_cuda.oc_cuda(
        multiple_gpu.beta,
        multiple_gpu.q,
        multiple_gpu.x,
        multiple_gpu.y,
        multiple_gpu.which_cond_point,
        multiple_gpu.row_splits,
        multiple_gpu.cond_row_splits,
        multiple_gpu.cond_indices,
        multiple_gpu.cond_counts,
    ).cpu()
    print(f'{losses_cuda=}')
    print(f'{losses_cpp=}')
    # Don't compare L_beta_cond_logterm and L_noise losses here
    assert torch.allclose(losses_cuda[:3], losses_cpp[:3])


def test_oc_python_gpu_batch():
    import torch_cmspepr

    print('Running CPU extension')
    losses_cpp = torch_cmspepr.oc(
        multiple.beta, multiple.q, multiple.x, multiple.y, multiple.batch
    )
    print('Running CUDA extension')
    losses_cuda = torch_cmspepr.oc(
        multiple_gpu.beta,
        multiple_gpu.q,
        multiple_gpu.x,
        multiple_gpu.y,
        multiple_gpu.batch,
    ).cpu()
    print(f'{losses_cuda=}')
    print(f'{losses_cpp=}')
    assert torch.allclose(losses_cuda, losses_cpp)
