# ruff: noqa: E402
import os.path as osp
import logging
import torch

__version__ = '1.1.0'


def setup_logger(name: str = "cmspepr") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = setup_logger()

from . import extensions as ext
LOADED_OPS = ext.LOADED_OPS

from torch_cmspepr.select_knn import select_knn, knn_graph
import torch_cmspepr.objectcondensation as objectcondensation
from torch_cmspepr.objectcondensation import oc, oc_noext, oc_noext_jit, calc_q_betaclip

__all__ = [
    'select_knn', 'knn_graph',
    'objectcondensation',
    'oc', 'oc_noext', 'oc_noext_jit', 'calc_q_betaclip',
    'logger'
    ]
