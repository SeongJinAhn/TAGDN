from re import X
from statistics import mode
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.model import TAGDN
import argparse
from src.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=256,
                        help='Node dimension d')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')  
    parser.add_argument('--num_layers', type=int, default=10,
                        help='number of layer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device') 
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='prob of teleport')
    parser.add_argument('--t', type=float, default=3,
                        help='diffusion time')
    parser.add_argument('--diffusion', type=str, default='ppr',
                        help='diffusion model [ppr / heat / gaussian]')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=2)

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    dev = args.device
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    if args.diffusion == 'ppr':
        coeff = args.alpha
    if args.diffusion == 'heat':
        coeff = args.t


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
    neg_edge = make_negative_edge(edge, neg_idx, node_features.size()[0]).cuda()

    labels = torch.LongTensor(labels).to(dev)
    train_idx = train_val_test_idx['train_idx']
    train_node = torch.LongTensor(np.sort(train_idx)).to(dev)
    val_idx = train_val_test_idx['val_idx']
    valid_node = torch.LongTensor(np.sort(val_idx)).to(dev)
    test_idx = train_val_test_idx['test_idx']
    test_node = torch.LongTensor(np.sort(test_idx)).to(dev)

    num_classes = max(labels).item()+1
    final_f1 = 0

    node_features = node_features.to(args.device)
    model = TAGDN(num_class=num_classes,
                        num_layers=num_layers,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        alpha=coeff,
                        type_nodes=type_nodes,
                        mode=args.diffusion,
                        dataset=args.dataset,
                        temperature=args.temperature,
                        dev=args.device).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss() 
    # Train & Valid & Test
    best_val_f1 = 0

    for i in range(epochs):
        model.zero_grad() 
        model.train() 
        z = model(node_features, edge)
        loss = model.node_classification(z, train_node, labels[train_node])

        print('Epoch:  ',i+1)
        print('Train - Loss: {}'.format(loss.detach().cpu().numpy())) 
        loss.backward()
        optimizer.step()
        model.eval()
            
        with torch.no_grad():
            z = model(node_features, edge, training=False)
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(z[valid_node].cpu().detach().numpy(), labels[valid_node].cpu().detach().numpy(), num_classes=num_classes, mode='valid')
            val_f1 = svm_macro_f1_list[1][0]
            
          
        if float(best_val_f1) < float(val_f1):
            best_val_f1 = val_f1
            best_z = z

            with torch.no_grad():
                svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(z[test_node].cpu().detach().numpy(), labels[test_node].cpu().detach().numpy(), num_classes=num_classes)
                z = z[:labels.size()[0]].detach().cpu().numpy()
                kmeans = KMeans(n_clusters=len(torch.unique(labels.cpu())), random_state=42).fit(z)
                nmi = normalized_mutual_info_score(labels.cpu(), kmeans.labels_)
                ari = adjusted_rand_score(labels.cpu(), kmeans.labels_)
                print(nmi, ari)
               
