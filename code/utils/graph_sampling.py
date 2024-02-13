import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborLoader, GraphSAINTEdgeSampler
from torch_geometric.sampler import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor, AddSelfLoops
from dataloader import BasicDataset
from time import time
from model import *
from sklearn.metrics import roc_auc_score
import random
import os
from dataclasses import astuple

# Sample BPR triplets
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

# Bipartite graph with random neighbors (bug fixed!)
def subgraph_batches_v3(A, unique_user_n, users, pos_items, neg_items, batch_size, K=3, batches=None):
    if isinstance(A, SparseTensor):
        edge_attr = A.storage.value()
        edge_index = torch.vstack([A.storage.row(), A.storage.col()])
        data = Data(x=torch.arange(A.size(0)).to(world.device), edge_index=edge_index, edge_attr=edge_attr)
    else:
        data = Data(x=torch.arange(A.shape[0]).to(A.device), edge_index=A.indices(), edge_attr=A.values())
    
    # *** Caution: Use pyg-lib for sampling acceleration here. May need to fix the bugs in the PyG library (neighbor_sampler.py). ***
    loader = NeighborLoader(data, num_neighbors=[world.config['num_neighbors']] * K, 
                            directed=True, num_workers=4, pin_memory=True)
    for i in range(0, len(users), batch_size):
        u_batch, pi_batch, ni_batch = users[i:i + batch_size], pos_items[i:i + batch_size], neg_items[i:i + batch_size]
        
        unique_users_batch, user_inv_idx = torch.unique(u_batch, return_inverse=True)
        unique_items_batch, item_inv_idx = torch.unique(torch.cat([pi_batch, ni_batch]), 
                                                        return_inverse=True)
        item_inv_idx += len(unique_users_batch)
        real_batch_size = min(batch_size, len(users) - i)
        pos_inv_idx, neg_inv_idx = item_inv_idx[:real_batch_size], item_inv_idx[real_batch_size:]
        root_nodes = torch.cat([unique_users_batch, unique_items_batch + unique_user_n]).cpu()
        
        batch_data = loader.filter_fn(loader.collate_fn(root_nodes))
        node_id = batch_data.x
        adj_t = SparseTensor(row=batch_data.edge_index[1].to(world.device), col=batch_data.edge_index[0].to(world.device), 
                             value=batch_data.edge_attr, sparse_sizes=(node_id.shape[0], node_id.shape[0]))
        
        if world.model_name in ['lgn-vr', 'lgn-gas', 'ltgnn']:
            # This is compatible with the change of PyG versions
            batch = batch_data.num_sampled_nodes if batch_data.batch is None else batch_data.batch
            yield node_id, user_inv_idx, pos_inv_idx, neg_inv_idx, adj_t, batch
        else:
            root_node_mask = torch.zeros_like(node_id, device=world.device).type(torch.bool)
            root_node_mask[:root_nodes.shape[0]] = True

            yield node_id, user_inv_idx, pos_inv_idx, neg_inv_idx, adj_t, root_node_mask
        