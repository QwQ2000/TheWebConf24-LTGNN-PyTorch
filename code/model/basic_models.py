"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
import dataloader
from dataloader import BasicDataset
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import os

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class EmbeddingTable(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(EmbeddingTable, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
    
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.emb_dim = int(self.latent_dim)

        self.e_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.emb_dim, sparse=True)
        self.e_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.emb_dim, sparse=True)
        
        if self.config['pretrain'] == 0:
            if self.config['emb_init'] == 'xavier':
                nn.init.xavier_uniform_(self.e_user.weight, gain=1)
                nn.init.xavier_uniform_(self.e_item.weight, gain=1)
            elif self.config['emb_init'] == 'normal':
                nn.init.normal_(self.e_user.weight, std=0.1)
                nn.init.normal_(self.e_item.weight, std=0.1)
            world.cprint('Embedding - use {} initilizer'.format(self.config['emb_init']))
        elif self.config['pretrain'] == 1:
            self.e_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.e_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('Embedding - use pretrained data')
        elif self.config['pretrain'] == -1:
            emb_file_name = f'mf-{world.dataset}-D{self.config["latent_dim_rec"]}.pth.tar'
            emb_dict = torch.load(os.path.join(world.FILE_PATH, emb_file_name))
            #self.e_user.weight.data.copy_(F.normalize(emb_dict['embedding_user.weight'], p=2., dim=-1))
            #self.e_item.weight.data.copy_(F.normalize(emb_dict['embedding_item.weight'], p=2., dim=-1))
            self.e_user.weight.data.copy_(emb_dict['embedding_user.weight'])
            self.e_item.weight.data.copy_(emb_dict['embedding_item.weight'])
            print('Embedding - use MF embeddings')
        
        if self.config['emb_transform'] == 'none':
            self.emb_t = nn.Identity()
        elif self.config['emb_transform'] == 'mlp':
            self.emb_t = nn.Sequential(
                            nn.Linear(self.emb_dim, self.latent_dim),
                            nn.ReLU(),
                            nn.Linear(self.latent_dim, self.latent_dim)
                         )
        elif self.config['emb_transform'] == 'single_mat':
            self.emb_t = nn.Linear(self.emb_dim, self.latent_dim, bias=False)
            #nn.init.normal_(self.emb_t.weight, std=0.1) + 1 / self.emb_dim

    @property
    def embedding_user(self):
        return self.emb_t(self.e_user.weight)

    @property
    def embedding_item(self):
        return self.emb_t(self.e_item.weight)
    
    def transform_l2_reg(self):
        reg = torch.zeros(1, device=world.device)
        for param in self.emb_t.parameters():
            reg += param.norm(2).pow(2)
        return 1/2 * reg

    def forward(self, id=None):
        if id is None:
            users_emb = self.e_user.weight
            items_emb = self.e_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            all_emb = self.emb_t(all_emb)
        else:
            all_emb = torch.zeros(id.shape[0], self.emb_dim, device=self.e_user.weight.device)
            user_mask = id < self.num_users
            item_mask = torch.logical_not(user_mask)
            all_emb[user_mask] = self.e_user(id[user_mask])
            all_emb[item_mask] = self.e_item(id[item_mask] - self.num_users)
        
        return all_emb

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.table = EmbeddingTable(self.config, self.dataset)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        all_emb = self.table.forward()
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    input(g_droped[f])
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.table.embedding_user[users]
        pos_emb_ego = self.table.embedding_item[pos_items]
        neg_emb_ego = self.table.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users)) + self.table.transform_l2_reg()
                        
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
