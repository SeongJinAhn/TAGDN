from ctypes.wintypes import SC_HANDLE
from operator import neg
from numpy.lib.type_check import real
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from src.layers import APPNP, Diffusion
from sklearn.metrics import roc_auc_score, average_precision_score

class HetGDC(nn.Module):
    def __init__(self, num_class, num_layers, w_in, w_out, alpha, type_nodes, mode, dataset, temperature, dev):
        super(HetGDC, self).__init__()
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        self.dataset = dataset
        self.dev = dev
        self.temperature = temperature

        self.type_nodes = type_nodes.long()
        self.loss = nn.CrossEntropyLoss()

        self.prop = APPNP(K=1, alpha=0)
        self.appnp = APPNP(K=num_layers, alpha=alpha)
        self.heat_diffusion = Diffusion(K=num_layers, alpha=alpha, laplacian=True)
        self.node_features = nn.Parameter(torch.FloatTensor(self.w_in, self.w_out).uniform_(-1.0, 1.0).to(dev))
        self.LP_loss = nn.BCEWithLogitsLoss(reduction='mean') 

        if mode == '1-hop':
            self.diffusion = self.prop
        if mode == 'rw':
            self.diffusion = self.appnp
        if mode == 'heat':
            self.diffusion = self.heat_diffusion

        if dataset in ['DBLP']:
            self.act = 'tanh'
        else:
            self.act = 'l2'

        self.num_node_types = self.type_nodes.size()[0]
        for i in range(self.num_node_types):
            if i == 0:
                self.node_type = i * self.type_nodes[i]
            else:
                self.node_type += i * self.type_nodes[i]

        self.linear = nn.Linear(self.w_in, w_out)
        self.linear1 = nn.Linear(w_out, num_class)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear1.weight, gain=1.414)

    def type_adaptive_normalization(self, H):
        for i in range(self.type_nodes.size()[0]):
            tmp_std, tmp_mean = torch.std_mean(H[self.type_nodes[i].nonzero()],0)
            if self.dataset in ['DBLP']:  # scaling factor
                tmp_std = tmp_std * torch.sqrt(self.type_nodes[i].sum())
            if i == 0:
                mean_node_type = tmp_mean
                std_node_type = tmp_std
            else:
                mean_node_type = torch.cat((mean_node_type, tmp_mean))
                std_node_type = torch.cat((std_node_type, tmp_std))

        tilde_H = (H - mean_node_type[self.node_type]) / std_node_type[self.node_type]
        return (mean_node_type, std_node_type), tilde_H

    def type_aware_encoding(self, X):
        H = self.linear(X)
        H = self.activation(H, self.act)
        return H

    def type_infomation_restoration(self, tilde_Z, mean_node_type, std_node_type):
        Z = tilde_Z * std_node_type[self.node_type] + mean_node_type[self.node_type]
        return Z

    def forward(self, X, edge_index, neg_edge):
        # encoding
        H = self.type_aware_encoding(X)
        type_information, tilde_H = self.type_adaptive_normalization(H)
        mean_node_type, std_node_type = type_information

        # Generalized Graph Diffusion
        tilde_Z = self.diffusion(tilde_H, edge_index)

        # add type information to diffused style representations
        Z = self.type_infomation_restoration(tilde_Z, mean_node_type, std_node_type)
        return Z, self.TIMReg(tilde_H, edge_index, neg_edge).cuda()

    #DBLP : temperature=1 K=10 alpha=0.1
    #ACM :  temperature=0.5 K=15 alpha=0.1 
    #IMDB : temperature=1  K=10 
    #FREEBASE : temperature=3

    def TIMReg(self, z, edge_index, neg_edge):
        z = F.normalize(z, p=2, dim=1)
        pred_pos = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        pred_neg = torch.sum(z[neg_edge[0]] * z[neg_edge[1]], dim=1)

        pos = torch.exp(pred_pos/self.temperature).sum()
        neg = torch.exp(pred_neg/self.temperature).sum()
        del pred_pos, pred_neg
        loss = -torch.log(pos / (pos+neg))
        del pos, neg
        return loss

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
        z = self.linear1(z)
        y = z[train_node]
        loss = self.loss(y, train_target)
        return loss
    
    def lp_loss(self, z, pos_edge, neg_edge):
        pred, target = self.lp_predict(z, pos_edge, neg_edge)
        loss = self.LP_loss(pred, target)
        del pred, target
        return loss, 0

    def lp_predict(self, z, pos_edge, neg_edge):
        pred_pos = torch.sum(z[pos_edge[0]] * z[pos_edge[1]], dim=1)
        pred_neg = torch.sum(z[neg_edge[0]] * z[neg_edge[1]], dim=1)

        pred = torch.cat((pred_pos, pred_neg))
        target = [1 for x in range(len(pos_edge[0]))] + [0 for x in range(len(neg_edge[0]))]
        target = torch.Tensor(target).cuda()
        del pred_pos, pred_neg
        return pred, target

    def lp_test(self, z, pos_edge, neg_edge):
        pred, y = self.lp_predict(z, pos_edge, neg_edge)
        pred = torch.sigmoid(pred)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)
