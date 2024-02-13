'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
import numpy as np

import random
import os

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-D{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-L{world.config['lightGCN_n_layers']}-D{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn-ns':
        file = f"lgn_ns-{world.dataset}-L{world.config['lightGCN_n_layers']}-D{world.config['latent_dim_rec']}-N{world.config['num_neighbors']}.pth.tar"
    elif world.model_name == 'lgn-vr':
        file = f"lgn_vr-{world.dataset}-L{world.config['lightGCN_n_layers']}-D{world.config['latent_dim_rec']}-N{world.config['num_neighbors']}.pth.tar"
    elif world.model_name == 'lgn-gas':
        file = f"lgn_gas-{world.dataset}-L{world.config['lightGCN_n_layers']}-D{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'ltgnn':
        file = f"ltgnn-{world.dataset}-L{world.config['lightGCN_n_layers']}-D{world.config['latent_dim_rec']}-N{world.config['num_neighbors']}-A{world.config['appnp_alpha']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)
