# data/loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class NpyDataset(Dataset):
    """
    Dataset for NPY files shaped (N, 2049), each row is one sequence.
    """
    def __init__(self, npy_path: str, indices):
        self.arr = np.load(npy_path, mmap_mode="r")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        x = self.arr[row_idx]
        return torch.from_numpy(x.astype(np.int64)), row_idx


def collate_npy(batch):
    tokens = [item[0] for item in batch]
    idxs = [item[1] for item in batch]
    return torch.stack(tokens, dim=0), torch.tensor(idxs, dtype=torch.long)


def build_dataloaders(
    npy_path: str,
    num_train_samples: int,
    batch_size_ekfac: int,
    batch_size_influence: int,
    rank: int,
    world_size: int,
    num_workers: int = 2,
    seq_length: int = None
):
    """
    Build dataloaders for EK-FAC estimation and influence scoring.
    """
    arr = np.load(npy_path, mmap_mode="r")
    total_samples, total_len = arr.shape
    if seq_length is not None:
        expected_len = seq_length + 1
        if total_len != expected_len:
            raise ValueError(f"NPY should have shape (N, {expected_len}), got (N, {total_len})")

    total_to_use = min(num_train_samples, total_samples)
    all_indices = list(range(total_to_use))

    ds_ekfac = NpyDataset(npy_path, all_indices)
    ds_infl = NpyDataset(npy_path, all_indices)

    sampler_ekfac = DistributedSampler(
        ds_ekfac, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    sampler_infl = DistributedSampler(
        ds_infl, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dl_ekfac = DataLoader(
        ds_ekfac,
        batch_size=batch_size_ekfac,
        sampler=sampler_ekfac,
        collate_fn=collate_npy,
        num_workers=num_workers,
        pin_memory=True
    )

    dl_infl = DataLoader(
        ds_infl,
        batch_size=batch_size_influence,
        sampler=sampler_infl,
        collate_fn=collate_npy,
        num_workers=num_workers,
        pin_memory=True
    )

    return dl_ekfac, dl_infl
