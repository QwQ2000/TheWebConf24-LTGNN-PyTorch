'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import torch.nn.functional as F
import random

from model import *
from torch_geometric.data import NeighborSampler

class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.lr_gamma = config['lr_gamma']

        if world.model_name in ['mf', 'lgn']:
            #self.opt = optim.SparseAdam([recmodel.embedding_user.weight, recmodel.embedding_item.weight], lr=self.lr)
            self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        else:
            #self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
            self.opt = optim.SparseAdam([recmodel.table.e_user.weight, recmodel.table.e_item.weight], lr=self.lr)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.lr_gamma)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def main_loss(users_emb, pos_emb, neg_emb):
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    if world.config['loss_type'] == 'bce':
        neg_labels = torch.zeros(neg_scores.size()).to(world.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction='none')
        
        pos_labels = torch.ones(pos_scores.size()).to(world.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction='none')

        loss = (pos_loss + neg_loss).mean()
    elif world.config['loss_type'] == 'bpr':
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    elif world.config['loss_type'] == 'margin':
        loss = torch.mean(torch.clamp(neg_scores - pos_scores + 1, min=0, max=None))

    return loss

def reg_loss(users_emb0, pos_emb0, neg_emb0):
    return (1/2) * (users_emb0.norm(2).pow(2) + 
                    pos_emb0.norm(2).pow(2) +
                    neg_emb0.norm(2).pow(2)) / float(users_emb0.shape[0])
