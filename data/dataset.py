import json
import warnings
import logging
import os
import io
import random
import pickle
from typing import List
import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class ImageNetDataset(Dataset):
    def __init__(self, data_root, item_processor, shuffle=True):
        self.data_root = data_root
        self.item_processor = item_processor

        self.data_list = []
        for dirname in os.listdir(data_root):
            for filename in os.listdir(os.path.join(data_root, dirname)):
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    self.data_list.append((os.path.join(data_root, dirname, filename), dirname))
        
        self.data_list = sorted(self.data_list)
        
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.data_list)
        
        logger = logging.getLogger(__name__)
        logger.info(f'Load ImageNet dataset, totally {len(self.data_list)} images.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = {
            'img_path': self.data_list[index][0],
            'label': self.data_list[index][1]
        }
        return self.item_processor(data_item)


class UltraChatDataset(IterableDataset):
    def __init__(self, data_root, item_processor, seed=0, shuffle=True, read_n_chunks=1, num_processes=1, process_rank=0):
        self.file_paths = []

        for filename in os.listdir(data_root):
            if filename.endswith('.jsonl'):
                self.file_paths.append(os.path.join(data_root, filename))
        
        self.file_paths = sorted(self.file_paths)
        if shuffle:
            self.rng = np.random.default_rng(seed)
            self.rng.shuffle(self.file_paths)
        
        self.seed = seed
        self.shuffle = shuffle
        self.read_n_chunks = read_n_chunks
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.item_processor = item_processor
    
    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self.num_processes
        shard_id = self.process_rank * num_workers + worker_id

        # generate infinite stream of data
        while True:
            for i in range(0, len(self.file_paths), self.read_n_chunks):
                data = []
                for offset in range(self.read_n_chunks):
                    path = self.file_paths[i + offset]
                    with open(path, 'r') as fp:
                        for line in fp:
                            data.append(line)
                
                max_num_files = len(data) // num_shards * num_shards
                if max_num_files < len(data):
                    max_num_files += num_shards
                data = data + data[:max_num_files-len(data)]
                data = data[shard_id:max_num_files:num_shards]

                for record in data:
                    yield self.item_processor(record)


class ToIterableDataset(IterableDataset):
    def __init__(self, dataset, num_processes=1, process_rank=0):
        """
        Convert map style dataset to iterable dataset and generate an infinite iteration stream.
        """
        self.dataset = dataset
        self.num_processes = num_processes
        self.process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self.num_processes
        shard_id = self.process_rank * num_workers + worker_id
        num_data = len(self.dataset)

        while True:
            for i in range(shard_id, num_data, num_shards):
                yield self.dataset[i]


class MixIterDataset(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], weights: List[float], seed: int = 0):
        self.datasets = [iter(dataset) for dataset in datasets]
        self.weights = weights
        self.rng = random.Random(seed)
    
    def __iter__(self):
        while True:
            (dataset,) = self.rng.choices(self.datasets, weights=self.weights, k=1)
            yield next(dataset)

