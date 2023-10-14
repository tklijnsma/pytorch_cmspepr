import torch
import torch_cmspepr.utils

def test_batch_to_row_splits():
    batch = torch.LongTensor([
        0, 0, 0,
        1, 1,
        2, 2, 2, 2
        ])
    expected = torch.LongTensor([0, 3, 5, 9])
    out = torch_cmspepr.utils.batch_to_row_splits(batch)
    assert torch.allclose(expected, out)
    assert out.dtype == batch.dtype
