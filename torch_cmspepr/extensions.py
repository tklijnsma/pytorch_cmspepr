"""
This module contains the interface to the C++ extensions.
If the extensions are not available, the interface is stubbed out.
"""
import os, os.path as osp
from typing import Tuple
import torch

PACKAGE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))

# Keep track of which ops were successfully loaded
LOADED_OPS = set()

# Load the extensions as ops
def load_ops(so_file):
    so_file = osp.join(PACKAGE_DIR, so_file)
    if not osp.isfile(so_file):
        # logger.error(f'Could not load op: No file {so_file}')
        pass
    else:
        torch.ops.load_library(so_file)
        LOADED_OPS.add(osp.basename(so_file))

load_ops("select_knn_cpu.so")
load_ops("select_knn_cuda.so")
load_ops("oc_cpu.so")
load_ops("oc_grad_cpu.so")
load_ops("oc_cuda.so")

# ___________________________________________________________________
# JIT compile the interface to the extensions.
# If the extensions are not available, a function is compiled that will raise an error.


if 'select_knn_cpu.so' in LOADED_OPS:

    @torch.jit.script
    def select_knn_cpu(
        x: torch.Tensor,
        row_splits: torch.Tensor,
        mask: torch.Tensor,
        k: int,
        max_radius: float,
        mask_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.select_knn_cpu.select_knn_cpu(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )

else:

    @torch.jit.script
    def select_knn_cpu(
        x: torch.Tensor,
        row_splits: torch.Tensor,
        mask: torch.Tensor,
        k: int,
        max_radius: float,
        mask_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception('CPU extension for select_knn not installed')


if 'select_knn_cuda.so' in LOADED_OPS:

    @torch.jit.script
    def select_knn_cuda(
        x: torch.Tensor,
        row_splits: torch.Tensor,
        mask: torch.Tensor,
        k: int,
        max_radius: float,
        mask_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.select_knn_cuda.select_knn_cuda(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )

else:

    @torch.jit.script
    def select_knn_cuda(
        x: torch.Tensor,
        row_splits: torch.Tensor,
        mask: torch.Tensor,
        k: int,
        max_radius: float,
        mask_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception('CUDA extension for select_knn not installed')



if 'oc_cuda.so' in LOADED_OPS:

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


if 'oc_cpu.so' in LOADED_OPS:

    @torch.jit.script
    def oc_cpu(beta, q, x, y, row_splits) -> torch.Tensor:
        return torch.ops.oc_cpu.oc_cpu(beta, q, x, y, row_splits)

else:

    @torch.jit.script
    def oc_cpu(beta, q, x, y, row_splits) -> torch.Tensor:
        raise Exception('CPU extension for oc not installed')



if 'oc_grad_cpu.so' in LOADED_OPS:

    @torch.jit.script
    def oc_grad_cpu(
        model_output,
        beta,
        q,
        y,
        which_cond_point,
        cond_point_count,
        row_splits,
        ) -> torch.Tensor:
        return torch.ops.oc_grad_cpu.oc_grad_cpu(
            model_output,
            beta,
            q,
            y,
            which_cond_point,
            cond_point_count,
            row_splits,
            )

else:

    @torch.jit.script
    def oc_grad_cpu(
        model_output,
        beta,
        q,
        y,
        which_cond_point,
        cond_point_count,
        row_splits,
        ) -> torch.Tensor:
        raise Exception('CPU extension for oc_grad not installed')
