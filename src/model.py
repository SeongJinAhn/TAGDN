import time
from ctypes.wintypes import SC_HANDLE
from operator import neg
from numpy.lib.type_check import real
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from src.layers import PPR, HeatKernel, Gaussian
from sklearn.metrics import roc_auc_score, average_precision_score

class TAGDN(nn.Module):
    def __init__(self, num_class, num_layers, w_in, w_hid, w_out, alpha, type_nodes, mode, dataset, temperature, dev):
        super(TAGDN, self).__init__()
        self.w_in = w_in
        self.w_hid = w_hid
        self.w_out = w_out
        self.num_layers = num_layers
        self.dataset = dataset
        self.dev = dev
        self.temperature = temperature

        self.type_nodes = type_nodes.long()
        self.loss = nn.CrossEntropyLoss()
        self.renormalized = False # Renormalization trick : A -> (A+I)
        self.symmetric = 'row'

        if dataset in ['ACM']:
            self.act = 'tanh'
            self.symmetric = 'sym'
        else:
            self.act = 'l2'

        self.ppr_diffusion = PPR(K=num_layers, alpha=alpha, renormalized=self.renormalized, symmetric=self.symmetric)
        self.heat_diffusion = HeatKernel(K=num_layers, alpha=alpha, symmetric=self.symmetric, laplacian=True)
        self.gaussian_diffusion = Gaussian(K=num_layers, alpha=alpha, laplacian=True)
        self.node_features = nn.Parameter(torch.FloatTensor(self.w_in, self.w_out).uniform_(-1.0, 1.0).to(dev))

        if mode == 'ppr':
            self.diffusion = self.ppr_diffusion
        if mode == 'heat':
            self.diffusion = self.heat_diffusion
        if mode == 'gaussian':
            self.diffusion = self.gaussian_diffusion


        self.num_node_types = self.type_nodes.size()[0]
        for i in range(self.num_node_types):
            if i == 0:
                self.node_type = i * self.type_nodes[i]
            else:
                self.node_type += i * self.type_nodes[i]

        self.type_specific_encoder = nn.Linear(w_in, w_hid)
        self.linear = nn.Linear(w_hid, w_out)
        self.classifier = nn.Linear(w_out, num_class)
        nn.init.xavier_normal_(self.type_specific_encoder.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)
        nn.init.xavier_normal_(self.classifier.weight, gain=1.414)

    def type_adaptive_normalization(self, H):
        for i in range(self.type_nodes.size()[0]):
            tmp_std, tmp_mean = torch.std_mean(H[self.type_nodes[i].nonzero()],0, unbiased=False)
            if self.dataset in ['DBLP', 'IMDB']:
                tmp_std = tmp_std * torch.sqrt(self.type_nodes[i].sum()) + 1e-9
            if i == 0:
                mean_node_type = tmp_mean
                std_node_type = tmp_std
            else:
                mean_node_type = torch.cat((mean_node_type, tmp_mean))
                std_node_type = torch.cat((std_node_type, tmp_std))

        return mean_node_type, std_node_type

    def type_aware_encoding(self, X):
        H = self.type_specific_encoder(X)
        H = self.activation(H, self.act)
        return H

    def forward(self, X, edge_index, training=True):
        # Type-Aware Encoder
        start_time = time.time()
        H = self.type_aware_encoding(X)

        mean_node_type, std_node_type = self.type_adaptive_normalization(H)
        tilde_H = (H - mean_node_type[self.node_type]) / std_node_type[self.node_type]
        tilde_Z = self.diffusion(tilde_H, edge_index)
        Z = tilde_Z * std_node_type[self.node_type] + mean_node_type[self.node_type]

        Z = self.linear(Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def activation(self, z, type):
        if type == 'sigmoid':
            return torch.sigmoid(z)
        if type == 'tanh':
            return torch.tanh(z)
        if type == 'relu':
            return torch.relu(z)
        if type == 'leaky':
            return F.leaky_relu(z, 0.3)
        if type == 'l2':
            return F.normalize(z, p=2, dim=1)
    
    def node_classification(self, z, train_node, train_target):
        z = self.classifier(z)
        y = z[train_node]
        loss = self.loss(y, train_target)
        del z
        return loss
