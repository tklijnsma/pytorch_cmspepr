from typing import Union
import torch

def batch_to_row_splits(batch: torch.LongTensor):
    device = batch.device
    n_events = batch.max() + 1
    counts = torch.zeros(n_events, dtype=torch.long, device=device)
    counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.long))
    counts = torch.cat((torch.zeros(1, dtype=torch.long, device=device), counts))
    row_splits = torch.cumsum(counts, 0)
    return row_splits
