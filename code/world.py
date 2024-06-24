'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

# *** Update with your local folder path!!!***
ROOT_PATH = '/home/qwq2000/TheWebConf24-LTGNN-PyTorch/'
# ********************************************
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
#sys.path.append(join(CODE_PATH, 'sources'))
sys.path.append(join(CODE_PATH, 'utils/sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['yelp2018', 'alibaba-ifashion', 'amazon3']
all_models  = ['mf', 'lgn', 'lgn-ns', 'lgn-vr', 'lgn-gas', 'ltgnn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['appnp_alpha'] = args.appnp_alpha
config['input_mix'] = args.input_mix
config['grad_mix'] = args.grad_mix
config['K_val'] = args.K_val
config['emb_transform'] = args.emb_transform
config['emb_init'] = args.emb_init
config['num_neighbors'] = args.num_neighbors
config['lr_gamma'] = args.lr_gamma
config['appnp_adjust_coeff'] = eval(args.appnp_adjust_coeff)
config['LTGNN_selected_Ks'] = eval(args.LTGNN_selected_Ks)

config['vr_update_interval'] = 1
config['loss_type'] = args.loss_type

config['emb_dropout'] = args.emb_dropout

device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
mem_device = torch.device('cpu') if args.memory_placement == 'cpu' else device
cpu_emb_table = args.emb_placement == 'cpu'

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
eval_interval = args.eval_interval
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
