import world
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import utils
import dataloader
from pprint import pprint
from utils import timer, subgraph_batches_v3
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from random import randint
from .basic_procs import test_one_batch
CORES = multiprocessing.cpu_count() // 2

def train_LightGCN_NS(dataset, model, opt, epoch, neg_k=1, w=None):
    model = model.train()

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    
    A = dataset.getSparseGraph()

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    sampling_func = subgraph_batches_v3
    # For GAS, only 1-hop neighbors are needed
    K = 1 if world.model_name == 'lgn-gas' else world.config['lightGCN_n_layers']

    for (batch_i,
         (nodes_batch, 
          user_inv_idx, pos_inv_idx, neg_inv_idx,
          adj_t, batch)) in tqdm(enumerate(sampling_func(A, 
                                                         dataset.n_user, 
                                                         users, 
                                                         posItems, 
                                                         negItems, 
                                                         batch_size=world.config['bpr_batch_size'],
                                                         K=K,
                                                         batches=None))):

        nodes_batch = nodes_batch.to(world.device)
        x = model.table(nodes_batch).to(world.device)
        
        if world.model_name in ['lgn-ns']:
            z_out = model(x, nodes_batch, adj_t)
        elif world.model_name in ['lgn-vr', 'lgn-gas', 'ltgnn']:
            # batch indicates the indexes of 1, 2, 3, ... hop neighbors
            batch = torch.LongTensor(batch).to(world.device)
            z_out = model(x, nodes_batch, adj_t, batch)
        
        user_out, pos_out, neg_out = z_out[user_inv_idx], z_out[pos_inv_idx], z_out[neg_inv_idx]
        main_loss = utils.main_loss(user_out, pos_out, neg_out)

        user_emb0, pos_emb0, neg_emb0 = x[user_inv_idx], x[pos_inv_idx], x[neg_inv_idx]
        reg_loss = utils.reg_loss(user_emb0, pos_emb0, neg_emb0)

        loss = main_loss + reg_loss * world.config['decay']
        aver_loss += loss.item()

        opt.zero_grad()

        loss.backward()

        opt.step()

        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', loss.cpu().item(), epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    
    if world.model_name == 'ltgnn' and epoch % world.config['vr_update_interval'] == 0:
        print('Computing full aggregation for all nodes...')
        model.update_memory()
    
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()

    return f"loss{aver_loss:.3f}-{time_info}"


def test_LTGNN(dataset, Recmodel, epoch, w=None, multicore=0):
    Recmodel.eval()

    Ks = world.config['LTGNN_selected_Ks']
    for idx, (users_emb, items_emb) in enumerate(Recmodel.test_inference(Ks)):
        test_with_embeddings(dataset, users_emb, items_emb, epoch, Ks[idx], w, multicore)

def test_with_embeddings(dataset, users_emb, items_emb, epoch, K_val=3, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    
    # eval mode with no dropout
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hit': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = F.sigmoid(torch.matmul(users_emb[batch_users], items_emb.t()))
            #rating = F.sigmoid(torch.matmul(users_emb[batch_users.long()], items_emb.t()))
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hit'] += result['hit']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hit'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}(K={K_val})',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}(K={K_val})',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}(K={K_val})',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Hit@{world.topks}(K={K_val})',
                          {str(world.topks[i]): results['hit'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        results['K_val'] = K_val
        print(results)
        return results