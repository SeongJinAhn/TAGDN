import time
import scipy
from ctypes.wintypes import SC_HANDLE
from operator import neg
from numpy.lib.type_check import real
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from src.layers import PPR, HeatKernel, Gaussian
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, average_precision_score


import torch.nn as nn
class RBF(nn.Module):
    
    def __init__(self, n_kernels=3, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = (torch.cdist(X, X) ** 2).cuda()
        del X
        tmp1 = -L2_distances[None, ...].cuda()
        tmp2 = (self.get_bandwidth(L2_distances).cuda() * self.bandwidth_multipliers.cuda()).cuda()
        del L2_distances
        loss = torch.exp(tmp1.cpu() /  tmp2[:, None, None].cpu()).sum(dim=0).cuda()
        del tmp1, tmp2
        return loss
    
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
        self.LP_loss = nn.BCEWithLogitsLoss(reduction='mean') 

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

        self.batch_norm = []
        for i in range(self.type_nodes.size()[0]):
            self.batch_norm.append(nn.BatchNorm1d(w_hid).cuda())

    def type_specific_statistics(self, H):
        for i in range(self.type_nodes.size()[0]):
            tmp_mean = torch.mean(H[self.type_nodes[i].nonzero()],0)
            tmp_std = 1 / torch.sqrt(self.type_nodes[i].sum()-1) * ((H[self.type_nodes[i].nonzero()] - tmp_mean) ** 2).sum(0)
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
        H = self.type_aware_encoding(X)
        
        mean_node_type, std_node_type = self.type_specific_statistics(H)
        tilde_H = (H - mean_node_type[self.node_type]) / std_node_type[self.node_type]
        loss = self.wasserstein_loss(tilde_H)
        tilde_H = F.dropout(tilde_H, p=0.5, training=training)
        tilde_Z = self.diffusion(tilde_H, edge_index)
        Z = tilde_Z * std_node_type[self.node_type] + mean_node_type[self.node_type]
 
        Z = F.normalize(Z, p=2, dim=1)
        return Z, loss

    def wasserstein_loss(self, embeddings):
        embeddings = embeddings.cpu().detach().numpy()
        embeddings = embeddings.flatten()
        gaussian_sample = np.random.normal(0, 1, embeddings.shape[0])
        distance = wasserstein_distance(gaussian_sample, embeddings)
        return distance


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
