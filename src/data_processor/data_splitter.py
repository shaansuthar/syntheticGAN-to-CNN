from torch.utils.data import Subset
import random


class DataSplitter:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_subset(self, indices):
        return Subset(self.dataset, indices)

    def split_indices(self, total_size, split_sizes):
        indices = list(range(total_size))
        random.shuffle(indices)
        splits = []
        start = 0
        for size in split_sizes:
            end = start + size
            splits.append(indices[start:end])
            start = end
        return splits
