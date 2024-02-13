import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import procedure as Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import networkx as nx
from torch_sparse import SparseTensor

Recmodel = register.get_model_class(world.model_name)(world.config, dataset)
if not world.cpu_emb_table:
    Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

t = time.time()

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # Training
    for epoch in range(world.TRAIN_epochs):
        if epoch % world.eval_interval == 0 and epoch != 0:
            cprint("[TEST]")
            if world.model_name in ['ltgnn']:
                Procedure.test_LTGNN(dataset, Recmodel, epoch, w, world.config['multicore'])
            else:
                if world.model_name == 'mf':
                    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                else:
                    Procedure.Test_accelerated(dataset, Recmodel, epoch, w, world.config['multicore'])
        
        # Train one epoch
        start = time.time()
        if world.model_name in ['lgn-ns', 'lgn-vr', 'lgn-gas', 'ltgnn']:
            output_information = Procedure.train_LightGCN_NS(dataset, Recmodel, bpr.opt, epoch, neg_k=Neg_k,w=w)
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        t = time.time() - start
        
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} time={t}')
        
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()
