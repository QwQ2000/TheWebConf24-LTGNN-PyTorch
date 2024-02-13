'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.0015,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=2e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='yelp2018',
                        help="available datasets: [yelp2018, alibaba-ifashion, amazon3]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1001)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    # 0: Initialize from scratch, 1: Use the pretrained embeddings, -1: Use the MF embeddings
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='ltgnn', help='rec-model, support [mf, lgn, lgn-ns, lgn-vr, lgn-gas, ltgnn]')

    parser.add_argument('--lr_gamma', type=float,default=1,
                        help="the exponential learning rate decay")
    parser.add_argument('--appnp_alpha', type=float,default=0.1)
    parser.add_argument('--input_mix', type=float,default=0.0)
    parser.add_argument('--grad_mix', type=float,default=0.0)
    parser.add_argument('--device', type=int,default=2)
    parser.add_argument('--K_val', type=int, default=3)
    parser.add_argument('--emb_transform', type=str, default='none', choices=['mlp', 'single_mat', 'none'])
    parser.add_argument('--emb_init', type=str, default='normal', choices=['normal', 'xavier'])
    parser.add_argument('--num_neighbors', type=int, default=15)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--loss_type', type=str, default='bpr', choices=['bpr', 'bce', 'margin'])
    parser.add_argument('--memory_placement', type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--emb_placement', type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--emb_dropout', type=float, default=0)
    parser.add_argument('--appnp_adjust_coeff', type=str, default='[0.2, 0, 0]')
    parser.add_argument('--LTGNN_selected_Ks', type=str, default='[3]')
    
    return parser.parse_args()
