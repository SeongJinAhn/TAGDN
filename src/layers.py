import math
from operator import neg
from os import remove
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing,GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, negative_sampling
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_max, scatter_add


def softmax_(src, index, num_nodes):
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class Diffusion(MessagePassing):
    def __init__(self, K, alpha, laplacian=True, bias=True, **kwargs):
        super(Diffusion, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.laplacian=laplacian
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, normalize_type='row',
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        if normalize_type == 'row':
            deg_inv_sqrt = deg.pow(-1)
        if normalize_type == 'sym':
            deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if normalize_type == 'row':
            return edge_index,  edge_weight * deg_inv_sqrt[col]
        if normalize_type == 'sym':
            return edge_index,  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index = to_undirected(edge_index, num_nodes=x.size()[0])
        edge_index = remove_self_loops(edge_index)[0]
        
        if self.laplacian == False:
            edge_index = add_self_loops(edge_index, num_nodes=x.size()[0])[0]
        
        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, normalize_type='row', dtype=x.dtype)
        if self.laplacian == True:
            edge_index, norm = add_self_loops(edge_index, norm, fill_value=-1,num_nodes=x.size()[0])

        orig = x
        for i in range(self.K, 0, -1):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.alpha * x / i + orig
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)

class APPNP(MessagePassing):
    def __init__(self, K, alpha, bias=True, **kwargs):
        super(APPNP, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, normalize_type='row',
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        if normalize_type == 'row':
            deg_inv_sqrt = deg.pow(-1)
        if normalize_type == 'sym':
            deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if normalize_type == 'row':
            return edge_index,  edge_weight * deg_inv_sqrt[col]
        if normalize_type == 'sym':
            return edge_index,  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_weight=None, laplacian=False):
        """"""
        edge_index = to_undirected(edge_index, num_nodes=x.size()[0])
        edge_index = remove_self_loops(edge_index)[0]       
        edge_index = add_self_loops(edge_index, num_nodes=x.size()[0])[0]
        
        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, normalize_type='row', dtype=x.dtype)
        orig = x

        for i in range(1, self.K+1):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = x * (1-self.alpha) + orig * self.alpha
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)