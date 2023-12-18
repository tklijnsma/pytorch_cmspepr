import torch
from torch_geometric.data import Data
import torch_cmspepr

import tqdm
import time


def make_random_event(n_nodes=10000, n_events=5):
    model_out = torch.rand((n_nodes, 32))

    # Varying event sizes
    event_fracs = torch.normal(torch.ones(n_events), 0.1)
    event_fracs /= event_fracs.sum()
    event_sizes = (event_fracs * n_nodes).type(torch.int)
    event_sizes[-1] += n_nodes - event_sizes.sum()  # Make sure it adds up to n_nodes

    batch = torch.arange(n_events).repeat_interleave(event_sizes)
    row_splits = torch.cat(
        (torch.zeros(1, dtype=torch.int), torch.cumsum(event_sizes, 0))
    )

    ys = []
    for i_event in range(n_events):
        # Somewhere between 3 and 8 particles
        n_clusters = torch.randint(3, 8, (1,)).item()
        cluster_fracs = torch.randint(50, 200, (n_clusters,)).type(torch.float)
        cluster_fracs[0] += 200  # Boost the amount of noise relatively
        cluster_fracs /= cluster_fracs.sum()
        cluster_sizes = (cluster_fracs * event_sizes[i_event]).type(torch.int)
        # Make sure it adds up to n_nodes in this event
        cluster_sizes[-1] += event_sizes[i_event] - cluster_sizes.sum()
        ys.append(torch.arange(n_clusters).repeat_interleave(cluster_sizes))
    y = torch.cat(ys)

    y = y.type(torch.int)
    row_splits = row_splits.type(torch.int)
    return model_out, y, batch, row_splits


def test_oc_performance():
    t_purepy = 0.0
    t_jit = 0.0
    t_cpp = 0.0
    N = 30

    # Run once to avoid adding compilation time to the benchmark
    try:
        torch_cmspepr.oc_noext()
    except Exception:
        pass

    for _ in tqdm.tqdm(range(N)):
        # Don't count prep work in performance
        model_out, y, batch, row_splits = make_random_event()
        data = Data(y=y.type(torch.long), batch=batch)
        beta = torch.sigmoid(model_out[:, 0]).contiguous()
        q = torch_cmspepr.calc_q_betaclip(torch.sigmoid(model_out[:, 0])).contiguous()
        x = model_out[:, 1:].contiguous()

        t0 = time.perf_counter()
        torch_cmspepr.oc_noext(beta, q, x, y, batch)
        t1 = time.perf_counter()
        torch_cmspepr.oc_noext_jit(beta, q, x, y, batch)
        t2 = time.perf_counter()
        torch_cmspepr.oc(beta, q, x, y, batch)
        t3 = time.perf_counter()

        t_purepy += t1 - t0
        t_jit += t2 - t1
        t_cpp += t3 - t2

    print(f'Average purepy time: {t_purepy/N:.4f}')
    print(f'Average jit time: {t_jit/N:.4f}')
    print(f'Average cpp time: {t_cpp/N:.4f}')
    print(f'Speed up is {t_purepy/t_jit:.2f}x (purepy vs. jit)')
    print(f'Speed up is {t_jit/t_cpp:.2f}x (jit vs. cpp)')


test_oc_performance()
