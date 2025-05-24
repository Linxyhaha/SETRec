import pandas as pd
import numpy as np
import torch
import time
import random

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen, n_query=1, n_sem=1):
        super(SequentialDataset, self).__init__()
        self.dataset = dataset
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

        self.n_query = n_query
        self.n_sem = n_sem

        train_dict = np.load(self.dataset + "training_dict.npy", allow_pickle=True).item()
        valid_dict = np.load(self.dataset + "validation_dict.npy", allow_pickle=True).item()
        test_dict = np.load(self.dataset + "testing_dict.npy", allow_pickle=True).item()

        for u_id, items in train_dict.items():
            self.n_user = max(self.n_user, u_id)
            self.m_item = max(self.m_item, max(items))
            if len(items) >= 2: # history should at least have two interactions
                train_items = items
                length = min(len(train_items), self.maxlen)
                for t in range(length):
                    self.trainData.append([train_items[-length:-length+t], train_items[-length+t]])
            else: 
                pass
            if len(valid_dict[u_id]):
                self.valData[u_id] = train_dict[u_id], valid_dict[u_id][0]
            
            if len(test_dict[u_id]):
                self.testData[u_id] = train_dict[u_id] + valid_dict[u_id], test_dict[u_id]
                self.testData[u_id] = [_+1 for _ in self.testData[u_id][0]], test_dict[u_id]

        
        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        seq = [_+1 for _ in seq]
        n_query = self.n_query
        return seq, label, n_query

    def __len__(self):
        return len(self.trainData)

class SequentialTestDataset(SequentialDataset):
    def __init__(self, dataset, maxlen, n_sample=2000, n_query=1, n_sem=1):
        super(SequentialTestDataset, self).__init__(dataset, maxlen, n_query, n_sem)
        self.n_sample = n_sample
        
        self.evalData = []
        for u_id in self.valData:
            self.evalData.append(self.valData[u_id])
        random.shuffle(self.evalData)
        self.evalData = self.evalData[:self.n_sample]

    def __getitem__(self, idx):
        seq, label = self.evalData[idx]
        seq = [_+1 for _ in seq]
        n_query = self.n_query
        return seq, label, n_query

    def __len__(self):
        return len(self.evalData)
    
@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        seqs, labels, n_query = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)

        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) * n_query[0] + [1] * len(seq) * n_query[0] for seq in seqs]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)
        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }
