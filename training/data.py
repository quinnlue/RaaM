
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import random
from torch.utils.data import Sampler
import random
from collections import defaultdict
import torch


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, tokens_per_batch, bucket_sizes, shuffle=True):
        self.shuffle = shuffle
        self.buckets = defaultdict(list)

        for idx, (_, _, bucket) in enumerate(dataset.row_maps):
            self.buckets[bucket].append(idx)

        self.batches = []
        for bucket, indices in self.buckets.items():
            bs = tokens_per_batch // bucket_sizes[bucket]
            for i in range(0, len(indices), bs):
                self.batches.append(indices[i:i+bs])

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)



class CoTDataset(Dataset):
    def __init__(self, data_root: str, batch_column: str = "bucket", data_column: str = "seq", ext=".parquet"):
        self.data_root = data_root
        self.ext = ext
        self.batch_column = batch_column
        self.data_column = data_column
        self.file_paths = self.get_file_paths()
        self.row_maps = []
        self.file_data = []
        self._build_row_maps()

    def get_file_paths(self):
        return sorted([f for f in os.listdir(self.data_root) if f.endswith(self.ext)])

    def _build_row_maps(self):
        """Build mapping from global index to (file_idx, row_idx, bucket)"""
        for file_idx, file_path in enumerate(self.file_paths):
            full_path = os.path.join(self.data_root, file_path)
            df = pd.read_parquet(full_path)
            self.file_data.append(df)
            for row_idx in range(len(df)):
                bucket = df.iloc[row_idx][self.batch_column]
                self.row_maps.append((file_idx, row_idx, bucket))

    def __getitem__(self, idx):
        file_idx, row_idx, _ = self.row_maps[idx]
        row = self.file_data[file_idx].iloc[row_idx]
        data = row[self.data_column]

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy()).long()  # .copy() to make writable
        elif isinstance(data, list):
            return torch.tensor(data)
        return data

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.row_maps)

    def __str__(self):
        return f"CoTDataset(data_root={self.data_root}, ext={self.ext}, file_paths={self.file_paths}, len={len(self)})"
