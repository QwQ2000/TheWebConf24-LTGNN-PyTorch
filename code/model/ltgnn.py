import world
import torch
import dataloader
from dataloader import BasicDataset
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from .basic_models import LightGCN
from .lightgcn import NSLightGCN

    
class APPNP(LightGCN):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(APPNP, self).__init__(config, dataset)
        self.alpha = config['appnp_alpha']
        self.adjust_coeff = torch.Tensor(config['appnp_adjust_coeff']).to(world.device)

        if self.adjust_coeff[-1] != 0:
            binary_g = torch.sparse_coo_tensor(indices=self.Graph.indices(), values=torch.ones_like(self.Graph.values()))
            deg = torch.sparse.sum(binary_g, dim=1).to_dense().unsqueeze(0)

            self.d_col_v = torch.sqrt(deg.T)
            self.d_row_v = torch.sqrt(deg) / binary_g.coalesce().values().shape[0]

    def computer(self):
        """
        propagate methods for APPNP
        """       
        input_emb = self.table.forward()

        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        z = input_emb
        for layer in range(self.n_layers):
            z = (1 - self.alpha) * torch.sparse.mm(g_droped, z) + self.alpha * input_emb
        
        light_out = z
        if self.adjust_coeff[0] != 0:
            light_out -= input_emb * self.adjust_coeff[0]
        if self.adjust_coeff[1] != 0:
            light_out -= torch.sparse.mm(g_droped, input_emb) * self.adjust_coeff[1]
        if self.adjust_coeff[2] != 0:
            inf_emb = self.d_col_v @ (self.d_row_v @ input_emb)
            light_out -= inf_emb * self.adjust_coeff[2]
        light_out /= (1 - torch.sum(self.adjust_coeff))

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class NSAPPNP(NSLightGCN):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NSAPPNP, self).__init__(config, dataset)
        self.alpha = config['appnp_alpha']
        self.adjust_coeff = torch.Tensor(config['appnp_adjust_coeff']).to(world.device) #config['appnp_adjust_coeff']

        if self.adjust_coeff[-1] != 0:
            binary_g = torch.sparse_coo_tensor(indices=self.Graph.indices(), values=torch.ones_like(self.Graph.values()))
            deg = torch.sparse.sum(binary_g, dim=1).to_dense().unsqueeze(0)

            self.d_col_v = torch.sqrt(deg.T)
            self.d_row_v = torch.sqrt(deg) / binary_g.coalesce().values().shape[0]
        
    def _compute_bias_norm(self, id, adj):
        # Compute degree normalization coefficients
        if self.config['num_neighbors'] != -1:
            n_neighbors = degree(adj.storage.row(), adj.size(0))
            n_neighbors[n_neighbors == 0] = 1
            bias_norm = self.deg[id] / n_neighbors
        else:
            bias_norm = torch.ones(id.shape[0]).to(world.device)
        
        return bias_norm
    
    def _appnp_coeff_adjust(self, z, input_emb, adj=None, id=None):
        light_out = z
        if self.adjust_coeff[0] != 0:
            light_out -= input_emb * self.adjust_coeff[0]
        if self.adjust_coeff[1] != 0:
            light_out -= adj @ input_emb * self.adjust_coeff[1]
        if self.adjust_coeff[2] != 0:
            if id is not None:
                inf_emb = self.d_col_v[id, :] @ (self.d_row_v[:, id] @ input_emb)
            else:
                inf_emb = self.d_col_v @ (self.d_row_v @ input_emb)
            light_out -= inf_emb * self.adjust_coeff[2]
        light_out /= (1 - torch.sum(self.adjust_coeff))

        return light_out

    def forward(self, x, id, adj, batch):
        bias_norm = self._compute_bias_norm(id, adj)
        
        input_emb = x
        z = x
        for _ in range(self.n_layers):
            z = (1 - self.alpha) * bias_norm.unsqueeze(1) * (adj @ z) + self.alpha * input_emb

        z = self._appnp_coeff_adjust(z, input_emb, adj, id)
        
        return z

    # Adapted from full-batch APPNP
    @torch.no_grad()
    def inference(self, K=None, coeff_adjust=True):  
        input_emb = self.table.forward().to(world.device)

        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        z = input_emb
        K = self.n_layers if K is None else K
        for layer in range(K):
            if isinstance(g_droped, SparseTensor):
                az = g_droped @ z
            else:
                az = torch.sparse.mm(g_droped, z)
            z = (1 - self.alpha) * az + self.alpha * input_emb
            
        if coeff_adjust:
            light_out = self._appnp_coeff_adjust(z, input_emb, g_droped)
        else:
            light_out = z
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class ForwardImplicitAPPNP(NSAPPNP):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(ForwardImplicitAPPNP, self).__init__(config, dataset)
        self.beta = self.config['input_mix']

        self.z_mem = None
        self._init_memory()

        self.K_val = self.config['K_val']
        self.prop = ForwardImplicitAPPNPLayer()
    
    def _init_memory(self, iteration=7):
        user_emb, item_emb = super(ForwardImplicitAPPNP, self).inference(iteration)
        self.z_mem = torch.vstack([user_emb, item_emb]).detach()
    
    def forward(self, x, id, adj, batch):
        bias_norm = self._compute_bias_norm(id, adj)
        
        input_emb = x
        z_mem = self.z_mem[id].to(world.device)
        z_out = self.prop.apply(x, adj, bias_norm, self.n_layers, self.alpha, self.beta, batch, z_mem)

        # Memory update
        self.z_mem[id[:batch[0]]] = z_out[:batch[0]].detach().to(self.z_mem)

        # Output
        z_out = self._appnp_coeff_adjust(z_out, input_emb, adj, id)

        return z_out

    # Directly use the fixed point for inference!
    @torch.no_grad()
    def inference(self, K=None, coeff_adjust=True):
        K = self.K_val if K is None else K
        if K != 0:
            return super(ForwardImplicitAPPNP, self).inference(K, coeff_adjust)

        input_emb = self.table.forward().to(world.device)
        z = self.z_mem.to(world.device)

        if coeff_adjust:
            light_out = self._appnp_coeff_adjust(z, input_emb, self.Graph)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    # Try different settings of inference layers (0 for using the stored fixed point)
    @torch.no_grad()
    def test_inference(self, selected_Ks):
        max_K = max(selected_Ks)
        if 0 in selected_Ks:
            yield self.inference(K=0)
        
        input_emb = self.table.forward().to(world.device)

        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        z = input_emb
        for layer in range(1, max_K + 1):
            if isinstance(g_droped, SparseTensor):
                az = g_droped @ z
            else:
                az = torch.sparse.mm(g_droped, z)
            z = (1 - self.alpha) * az + self.alpha * input_emb
            if layer in selected_Ks:
                light_out = self._appnp_coeff_adjust(z, input_emb, self.Graph)
                yield torch.split(light_out, [self.num_users, self.num_items])
    

class ForwardImplicitAPPNPLayer(torch.autograd.Function):

    @staticmethod
    def forward(self, x, adj, bias_norm, K, alpha, beta, batch, z_mem, **kwargs):
        
        self.alpha = alpha
        self.K = K
        self.adj = adj
        
        self.save_for_backward(bias_norm, batch)
        
        input_emb = x
        z = (1 - beta) * z_mem + beta * x

        # Only use one propagation layer in forward process
        z = (1 - self.alpha) * bias_norm.unsqueeze(1) * (self.adj @ z) + self.alpha * input_emb

        return z

    @staticmethod
    def backward(self, grad_output):
        bias_norm, batch = self.saved_tensors

        input_grad = grad_output
        g = grad_output
        for _ in range(self.K):
            g = (1 - self.alpha) * bias_norm.unsqueeze(1) * (self.adj @ g) + self.alpha * input_grad
        
        # Remove the inexact gradients
        #g[batch[0]:] = 0
        return g, None, None, None, None, None, None, None


class ImplicitAPPNP(ForwardImplicitAPPNP):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(ImplicitAPPNP, self).__init__(config, dataset)
        self.theta = self.config['grad_mix']

        N = self.num_users + self.num_items
        # Initialize z_memory with zero instead of 7-layer APPNP result
        #self.z_mem = torch.zeros(N, self.table.latent_dim).to(world.mem_device)
        self.g_mem = torch.zeros(N, self.table.latent_dim).to(world.mem_device)

        self.prop = ImplicitAPPNPLayer()
    
    def forward(self, x, id, adj, batch):
        bias_norm = self._compute_bias_norm(id, adj)

        # Use x + 0 to create a new node in the torch computation graph
        input_emb, implicit_input = x, x + 0

        # Gradient memory update hook
        def backward_hook(grad):            
            self.g_mem[id[:batch[0]]] = (1 - torch.sum(self.adjust_coeff)) * grad[:batch[0]].detach().to(self.g_mem)
            return grad
        
        # Only the gradients of the implicit layers are saved in memory
        #x.register_hook(backward_hook)
        implicit_input.register_hook(backward_hook)

        z_mem, g_mem = self.z_mem[id].to(world.device), self.g_mem[id].to(world.device)
        z_out = self.prop.apply(implicit_input, adj, bias_norm, self.n_layers, self.alpha, 
                                self.beta, self.theta, batch, z_mem, g_mem)
        
        self.z_mem[id[:batch[0]]] = z_out[:batch[0]].detach().to(self.z_mem)

        # Output
        z_out_adjust = self._appnp_coeff_adjust(z_out, input_emb, adj, id)
        
        return z_out_adjust


class ImplicitAPPNPLayer(torch.autograd.Function):

    @staticmethod
    def forward(self, x, adj, bias_norm, K, alpha, beta, theta, batch, z_mem, g_mem, **kwargs):
        
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.adj = adj
        
        self.save_for_backward(g_mem, bias_norm, batch)
        
        input_emb = x
        if torch.equal(z_mem[:batch[0]], torch.zeros_like(z_mem)[:batch[0]].to(world.device)):
            z = x
        else:
            z = (1 - beta) * z_mem + beta * x

        #for _ in range(self.K):
        z = (1 - self.alpha) * bias_norm.unsqueeze(1) * (self.adj @ z) + self.alpha * input_emb

        return z

    @staticmethod
    def backward(self, grad_output):
        g_mem, bias_norm, batch = self.saved_tensors

        input_grad = grad_output
        if torch.equal(g_mem[:batch[0]], torch.zeros_like(g_mem)[:batch[0]].to(world.device)):
            g = grad_output
        else:
            g = (1 - self.theta) * g_mem + self.theta * grad_output
        
        for _ in range(self.K):
            g = (1 - self.alpha) * bias_norm.unsqueeze(1) * (self.adj @ g) + self.alpha * input_grad
        
        # Remove the inexact gradients
        #g[batch[0]:] = 0
        return g, None, None, None, None, None, None, None, None, None
    

class LTGNN(ImplicitAPPNP):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LTGNN, self).__init__(config, dataset)

        # Initialize VR memory
        self.update_memory()

        self.prop = VRImplicitAPPNPLayer()
        self.prev1 = None

    @torch.no_grad()
    def update_memory(self):
        if self.beta == 0:
            if self.n_layers == 1:
                # Single layer VR code
                self.in_mem = self.z_mem.clone().detach()
                self.in_mem.requires_grad_(False)

                A = self.Graph
                if isinstance(A, SparseTensor):
                    az = A @ self.in_mem
                else:
                    az = torch.sparse.mm(A, self.in_mem)
                
                self.in_aggr_mem = az.clone().detach()
                self.in_aggr_mem.requires_grad_(False)
            else:
                # Multi layer VR code

                # Space assignment
                N = self.num_items + self.num_users
                self.in_mem = torch.empty(self.n_layers, N, self.table.emb_dim).to(world.device)
                self.in_aggr_mem = torch.empty(self.n_layers, N, self.table.emb_dim).to(world.device)

                # Initialization
                A = self.Graph
                self.in_mem[0, :, :] = self.z_mem.clone().detach()

                # Propagation
                for i in range(self.n_layers):
                    if isinstance(A, SparseTensor):
                        in_aggr = A @ self.in_mem[i]
                    else:
                        in_aggr = torch.sparse.mm(A, self.in_mem[i])
                    self.in_aggr_mem[i, :, :] = in_aggr.clone().detach()
                    if i != self.n_layers - 1:
                        in_next = (1 - self.alpha) * in_aggr + self.alpha * self.in_mem[0]
                        self.in_mem[i + 1, :, :] = in_next.clone().detach()

                # No gradient for memory space
                self.in_mem.requires_grad_(False)
                self.in_aggr_mem.requires_grad_(False)
        else:
            raise NotImplementedError('Not Implemented beta != 0')
    
    def forward(self, x, id, adj, batch, debug=False):
        bias_norm = self._compute_bias_norm(id, adj)

        # Use x + 0 to create a new node in the torch computation graph
        input_emb, implicit_input = x, x + 0

        # Gradient memory update hook
        def backward_hook(grad):
            self.g_mem[id[:batch[0]]] = (1 - torch.sum(self.adjust_coeff)) * grad[:batch[0]].detach().to(self.g_mem)
            return grad
        implicit_input.register_hook(backward_hook)

        z_mem, g_mem = self.z_mem[id].to(world.device), self.g_mem[id].to(world.device)
        with torch.no_grad():
            if self.n_layers == 1:
                in_mem, in_aggr_mem = self.in_mem[id].to(world.device), self.in_aggr_mem[id].to(world.device)
            else:
                in_mem, in_aggr_mem = self.in_mem[:, id, :].to(world.device), self.in_aggr_mem[:, id, :].to(world.device)
        z_out = self.prop.apply(implicit_input, in_mem, in_aggr_mem, adj, bias_norm, self.n_layers, 
                                self.alpha, self.beta, self.theta, batch, z_mem, g_mem)
        
        if debug:
            def grad_debug_hook(grad):
                self.debug_grad = grad.detach()
                return grad
            z_out.register_hook(grad_debug_hook)
        
        self.z_mem[id[:batch[0]]] = z_out[:batch[0]].detach().to(self.z_mem)

        # Output
        z_out_adjust = self._appnp_coeff_adjust(z_out, input_emb, adj, id)

        if debug:
            return z_out, z_out_adjust

        return z_out_adjust


class VRImplicitAPPNPLayer(torch.autograd.Function):

    @staticmethod
    def forward(self, x, in_mem, in_aggr_mem, adj, bias_norm, K, alpha, beta, theta, batch, z_mem, g_mem, **kwargs):
        
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.adj = adj
        
        self.save_for_backward(g_mem, bias_norm, batch)
        
        input_emb = x
        if torch.equal(z_mem[:batch[0]], torch.zeros_like(z_mem)[:batch[0]].to(world.device)):
            z = x
        else:
            z = (1 - beta) * z_mem + beta * x

        assert beta == 0

        if self.K == 1:
            az = bias_norm.unsqueeze(1) * (self.adj @ (z - in_mem)) + in_aggr_mem
            z = (1 - self.alpha) * az + self.alpha * input_emb
        else:
            for i in range(K):
                az = bias_norm.unsqueeze(1) * (self.adj @ (z - in_mem[i, :, :])) + in_aggr_mem[i, :, :]
                z = (1 - self.alpha) * az + self.alpha * input_emb
        return z

    @staticmethod
    def backward(self, grad_output):
        g_mem, bias_norm, batch = self.saved_tensors

        input_grad = grad_output
        if torch.equal(g_mem[:batch[0]], torch.zeros_like(g_mem)[:batch[0]].to(world.device)):
            g = grad_output
        else:
            g = (1 - self.theta) * g_mem + self.theta * grad_output
        
        for _ in range(self.K):
            g = (1 - self.alpha) * bias_norm.unsqueeze(1) * (self.adj @ g) + self.alpha * input_grad
        
        # Remove the inexact gradients
        #g[batch[0]:] = 0
        return g, None, None, None, None, None, None, None, None, None, None, None