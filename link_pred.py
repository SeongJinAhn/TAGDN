from re import X
from statistics import mode
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.model_link import HetGDC
import argparse
from torch_geometric.utils import negative_sampling
from src.utils import load_DBLP_data, load_IMDB_data, load_ACM_data, load_FREEBASE_data, make_negative_edge, preprocess_attributes, preprocess_edges, divide_train_val_test_edge
from torch_geometric.utils import to_undirected

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg') 
    parser.add_argument('--num_layers', type=int, default=15,
                        help='number of layer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device') 
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='prob of teleport')
    parser.add_argument('--mode', type=str, default='rw',
                        help='diffusion model [1-hop / rw / heat')
    parser.add_argument('--temperature', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=0.01)

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    dev = args.device
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    if args.dataset == 'DBLP':
        features_list, adjM, labels, train_val_test_idx = load_DBLP_data()
        num_relation = 6
    if args.dataset == 'IMDB':
        features_list, adjM, labels, train_val_test_idx = load_IMDB_data()
        num_relation = 4
    if args.dataset == 'ACM':
        features_list, adjM, labels, train_val_test_idx = load_ACM_data()
        num_relation = 4
    if args.dataset == 'FREEBASE':
        features_list, adjM, labels, train_val_test_idx = load_FREEBASE_data([20, 40, 60], [3492, 2502, 33401, 4459])
        num_relation = 6
        
    N = adjM.shape[0]
    node_features = preprocess_attributes(N, args.dataset, features_list, dev)
    edge, neg_idx, type_nodes = preprocess_edges(N, adjM, args.dataset, features_list, dev)
    neg_edge = make_negative_edge(edge, neg_idx, node_features.size()[0], edge[0].size()[0] * 10)
    (train_edge, val_edge, test_edge), (train_neg_edge, val_neg_edge, test_neg_edge) = divide_train_val_test_edge(edge, neg_edge)
    node_features = node_features.to(args.device)

    model = HetGDC(num_class=1,
                        num_layers=num_layers,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        alpha=args.alpha,
                        type_nodes=type_nodes,
                        mode=args.mode,
                        dataset=args.dataset,
                        temperature=args.temperature,
                        dev=args.device).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss() 
    best_val = 0
    # Train & Valid & Test
        
    for i in range(epochs):
        model.zero_grad() 
        model.train() 
    
        train_neg_edge = train_neg_edge[:, train_neg_edge[0] < train_neg_edge[1]]
        index = torch.randperm(train_neg_edge.size()[1])[:train_edge.size()[1]*30]
        batch_train_neg_edge = train_neg_edge[:,index]
        batch_train_neg_edge = to_undirected(batch_train_neg_edge)

        z, reg_loss = model(node_features, train_edge, batch_train_neg_edge)
        loss, _ = model.lp_loss(z, train_edge, batch_train_neg_edge)
        loss += reg_loss * args.gamma

        print('Epoch:  ',i+1)
        print('Train - Loss: {}'.format(loss.detach().cpu().numpy()))
        loss.backward()
        optimizer.step()
        model.eval()
            
        with torch.no_grad():
            z, reg_loss = model(node_features, train_edge, batch_train_neg_edge)
            val_auc, val_ap = model.lp_test(z, val_edge, val_neg_edge)
            test_auc, test_ap = model.lp_test(z, test_edge, test_neg_edge)

            if best_val < val_auc:
                best_val = val_auc
                selected_auc, selected_ap = test_auc, test_ap
                print('AUC : ',test_auc, 'AP : ', test_ap)
print('Completed')
print('AUC : ',selected_auc, 'AP : ', selected_ap)
