from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from random import sample, randint
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset


def make_interactions(
    m: int = 1_000_000, 
    n: int = 1_000, 
    max_j_per_i: int = 4
):
    interactions = []

    for i in range(m):
        for j in sample(range(n), randint(1, max_j_per_i)):
            interactions.append([i, j])

    return interactions

def make_interactions_train_test(
    m: int = 1_000_000, 
    n: int = 1_000, 
    max_j_per_i: int = 4
):
    m_split = m // 2; n_split = n // 2
    interactions_train, interactions_test = [], []
    
    for i in range(m):
        for j in sample(range(n), randint(1, max_j_per_i)):
            if i <= m_split or j <= n_split:                    
                interactions_train.append([i, j])
            else: 
                interactions_test.append([i, j])

    return (interactions_train, interactions_test)


def load_data(csv_path: str):
    """
    Expects a CSV with _ columns: [_, _]
    Returns: 
        - coords: numpy array of shape (_, _) with integer-coded (_, ...)
        - num_...: int
    """
    df = pd.read_csv(csv_path, encoding='utf-16', sep=',')
    print(f'There are {len(df)} _-_ pairs.') # co-occurrences

    # Build integer encodings
    us = sorted(df.iloc[:, 0].unique().tolist())
    vs = sorted(df.iloc[:, 1].unique().tolist())
    us2ix = {u: i for i, u in enumerate(us)}
    vs2ix = {v: j for j, v in enumerate(us)}
    num_m = len(us2ix); num_n = len(vs2ix)
    print(f'There are {num_m} m and {num_n} n')

    # Map to ints
    df.iloc[:, 0] = df.iloc[:, 0].map(us2ix).astype('int64')
    df.iloc[:, 1] = df.iloc[:, 1].map(vs2ix).astype('int64')

    # Sort and return as numpy
    df.sort_values(by=df.columns.tolist(), inplace=True, kind='mergesort')
    interactions = df.values.astype(np.int64)

    return interactions, num_m, num_n


def build_dictionary(interactions_list: list):
    D = defaultdict(set)

    for i_list in interactions_list: 
        for i, j in i_list:
            D[i].add(j)
    
    return D


class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]


def collate_fn(batch, D, num_n, negatives):    
    U_, A = zip(*batch)

    U, S, y = [], [], []

    for u, a in zip(U_, A):
        U.append(u)
        S.append(a)
        y.append(1)

        for _ in range(negatives):
            while True:
                b_ = randint(0, num_n - 1)
                if b_ not in D[u]:
                    U.append(u)
                    S.append(b_)
                    y.append(0)
                    break

    return (
        torch.tensor(U, dtype=torch.long),
        torch.tensor(S, dtype=torch.long),
        torch.tensor(y, dtype=torch.float),
    )


def build_loader(
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = True,
        collate_fn = None,
        collate_kwargs: dict | None = None,
        device = None
):    
    if collate_fn is not None and collate_kwargs:
        collate_fn = partial(collate_fn, **collate_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size // (1 + collate_kwargs['negatives']),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=device == 'cuda',
        collate_fn=collate_fn
    )


def regularized_loss(
    logits: torch.Tensor, 
    target: torch.Tensor, 
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    device: torch.device,
    k: int, 
    lambda_: float = 1e-5
): 
    loss = binary_cross_entropy_with_logits(logits, target)

    eye = torch.eye(k, device=device) 
    U_l = torch.norm(U.t() @ U - eye, p='fro')
    V_l = torch.norm(V.t() @ V - eye, p='fro')
    regularizer = lambda_ * (U_l + V_l)

    return (loss, regularizer)