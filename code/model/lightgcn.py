import world
import torch
import dataloader
from dataloader import BasicDataset
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from .basic_models import *


class NSLightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NSLightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']

        self._init_weight()
    
    def _init_weight(self):
        self.table = EmbeddingTable(self.config, self.dataset)
        self.Graph = self.dataset.getSparseGraph()
        
        # The degree of each node
        binary_g = torch.sparse_coo_tensor(indices=self.Graph.indices(), values=torch.ones_like(self.Graph.values()))
        self.deg = torch.sparse.sum(binary_g, dim=1).to_dense()
        
        # Could be faster? torch.sparse.FloatTensor v.s. torch_sparse.SparseTensor
        # self.Graph = SparseTensor(row=self.Graph.indices()[0], col=self.Graph.indices()[1], value=self.Graph.values())

    def forward(self, x, id, adj):
        # Compute degree normalization coefficients
        if self.config['num_neighbors'] != -1:
            n_neighbors = degree(adj.storage.row(), adj.size(0))
            n_neighbors[n_neighbors == 0] = 1
            bias_norm = self.deg[id] / n_neighbors
        else:
            bias_norm = torch.ones(id.shape[0]).to(world.device)
        
        embs = [x]
        z = x
        for _ in range(self.n_layers):
            z = bias_norm.unsqueeze(1) * (adj @ z)
            embs.append(z)
        embs = torch.stack(embs, dim=1)
        z_out = torch.mean(embs, dim=1)

        return z_out

    # Adapted from the original implementation of LightGCN
    @torch.no_grad()
    def inference(self):  
        all_emb = self.table.forward()
        embs = [all_emb]

        g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if isinstance(g_droped, SparseTensor):
                all_emb = g_droped @ all_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


class VRLightGCN(NSLightGCN):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(VRLightGCN, self).__init__(config, dataset)
        
        # For slicing operations
        self.Graph = SparseTensor(row=self.Graph.indices()[0], col=self.Graph.indices()[1], value=self.Graph.values())

        N = self.num_users + self.num_items
        self.memory = torch.zeros(self.n_layers, N, self.table.latent_dim).to(world.mem_device)
    
    def forward(self, x, id, adj, batch):
        # Compute degree normalization coefficients
        if self.config['num_neighbors'] != -1:
            n_neighbors = degree(adj.storage.row(), adj.size(0))
            n_neighbors[n_neighbors == 0] = 1
            bias_norm = self.deg[id] / n_neighbors
        else:
            bias_norm = torch.ones(id.shape[0]).to(world.device)
        
        embs = [x]
        z = x
        for l in range(self.n_layers):
            # Variance reduction
            delta_z = bias_norm.unsqueeze(1) * (adj @ (z - self.memory[l, id, :]))
            z_full = self.Graph[id, :] @ self.memory[l, :, :]
            z = delta_z + z_full
            embs.append(z)
        
        # Memory update
        with torch.no_grad():
            for l in range(self.n_layers):
                if l == 0:
                    self.memory[l, id, :] = embs[l].detach()
                else:
                    node_max_idx = torch.sum(batch[:-l])
                    self.memory[l, id[:node_max_idx], :] = embs[l][:node_max_idx].detach()
        
        embs = torch.stack(embs, dim=1)
        z_out = torch.mean(embs, dim=1)
        
        return z_out


class GASLightGCN(NSLightGCN):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(GASLightGCN, self).__init__(config, dataset)
        
        N = self.num_users + self.num_items
        self.memory = torch.zeros(self.n_layers - 1, N, self.table.latent_dim).to(world.mem_device)
        self.__memory_initialize()
    
    @torch.no_grad()
    def __push_and_pull(self, x, id, layer, batch_size):
        self.memory[layer, id[:batch_size], :] = x[:batch_size].detach()
        x[batch_size:] = self.memory[layer, id[batch_size:], :]
        return x
    
    @torch.no_grad()
    def __memory_initialize(self):
        print('Initializing LightGCN-GAS memory embeddings...')
        all_emb = self.table.forward().to(world.device)

        g_droped = self.Graph
        for layer in range(self.n_layers - 1):
            if isinstance(g_droped, SparseTensor):
                all_emb = g_droped @ all_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            self.memory[layer, :, :] = all_emb.detach()
        
    def forward(self, x, id, adj, batch):
        embs = [x]
        z = x
        for l in range(self.n_layers - 1):
            z = adj @ z
            z = self.__push_and_pull(z, id, l, batch[0])
            embs.append(z)
        embs.append(adj @ z)

        embs = torch.stack(embs, dim=1)
        z_out = torch.mean(embs, dim=1)
        
        return z_out
