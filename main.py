from re import X
from statistics import mode
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.model import HetGDC
import argparse
from torch_geometric.utils import negative_sampling
from src.utils import load_DBLP_data, load_IMDB_data, load_ACM_data, load_freebase, evaluate_results_nc, make_edge, evaluate_results_nc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FREEBASE',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=256,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg') 
    parser.add_argument('--num_layers', type=int, default=10,
                        help='number of layer')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device') 
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='prob of teleport')
    parser.add_argument('--mode', type=str, default='heat',
                        help='diffusion model [1-hop / rw / heat')

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
        features_list, adjM, labels, train_val_test_idx = load_freebase([20, 40, 60], [3492, 2502, 33401, 4459])
        num_relation = 6
        
    num_nodes = adjM.shape[0]
    index = adjM.nonzero()

    total_features = 0
    for i in range(len(features_list)):
        total_features += features_list[i].shape[1]
        if args.dataset in ['ACM', 'IMDB']:
            continue

    node_features = torch.FloatTensor(num_nodes, total_features).to(dev)
    cnt1,cnt2 = 0,0
    for i in range(len(features_list)):
        edge_index = features_list[i].nonzero()
        if args.dataset == 'FREEBASE':
            edge_index = edge_index.t()
        node_features[cnt1+edge_index[0], cnt2+edge_index[1]] = torch.FloatTensor(features_list[i][edge_index[0], edge_index[1]]).to(dev)
        cnt1 += features_list[i].shape[0]
        if args.dataset in ['ACM', 'IMDB']:
            continue
        cnt2 += features_list[i].shape[1]           

    ############################
    src = torch.LongTensor([]).to(dev)
    dst = torch.LongTensor([]).to(dev)
    length = []
    node_types = []
    type_of_edges = torch.LongTensor([]).to(dev)
    ############################
    
    cnt1 = 0
    cnt = 0
    edges = []
    for i in range(len(features_list)):
        cnt2 = 0
        for j in range(len(features_list)):
            if adjM[cnt1:cnt1+features_list[i].shape[0], cnt2:cnt2+features_list[j].shape[0]].sum() != 0:           # 0 : 4057(A) -> 14328(P)
                tmp = adjM[cnt1:cnt1+features_list[i].shape[0], cnt2:cnt2+features_list[j].shape[0]].nonzero()      # 1 : 14328(P) -> 4057(A)  
                tmp = torch.LongTensor(tmp).to(dev)                                                              # 2 : 14328(P) -> 7723(T)
                src = torch.cat((src, tmp[0]+cnt1))                                                                 # 3 : 14328(P) -> 20(V)
                dst = torch.cat((dst, tmp[1]+cnt2))                                                                 # 4 : T -> P
                type_of_edges = torch.cat((type_of_edges, torch.ones_like(tmp[0]).to(dev)*cnt))                      # 5 : V -> P
                length.append(tmp.size()[1])

                node_types.append(set((tmp[0]+cnt1).tolist()))                                                             # 0 : M -> D
                node_types.append(set((tmp[1]+cnt2).tolist()))                                                             # 1 : M -> A
                
                cnt+=1                                                                                  

            cnt2 += features_list[j].shape[0]
        cnt1 += features_list[i].shape[0]


    for i,edge in enumerate(edges):
        if (args.bidirected == True) and i%2 == 1:
            continue

        tmp2 = torch.LongTensor(edges[i].nonzero()).to(dev)
        src2 = torch.cat((src2, tmp2[0]))
        dst2 = torch.cat((dst2, tmp2[1]))
        type_of_edges2 = torch.cat((type_of_edges2, torch.ones_like(tmp2[0])*i))
        length.append(tmp2.size()[1])

        node_types.append(set(tmp2[0].tolist()))
        node_types.append(set(tmp2[1].tolist()))

    a = set()
    type_nodes = torch.Tensor([]).to(dev)
    for i in range(len(node_types)):
        if list(node_types[i])[0] not in a:
            tmp = torch.zeros(num_nodes).to(dev)
            tmp[list(node_types[i])] = 1
            type_nodes = torch.cat((type_nodes, tmp.unsqueeze(0)))
        a = a | node_types[i]

    src, dst = src.to(dev), dst.to(dev)
    cnt = 0


    edge_type = torch.zeros(num_relation, sum(length)).to(dev)
    for i in range(num_relation):
        edge_type[i][cnt:cnt+length[i]] = 1
        cnt += length[i]

    labels = torch.LongTensor(labels).to(dev)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    train_node = torch.LongTensor(train_idx).to(dev)
    train_target = labels[train_node]
    valid_node = torch.LongTensor(val_idx).to(dev)
    valid_target = labels[valid_node]
    test_node = torch.LongTensor(test_idx).to(dev)
    test_target = labels[test_node]

    index = np.array([i for i in range(src.size()[0])])
    
    num_classes = torch.max(train_target).item()+1
    final_f1 = 0

    edges_split = {}
    edges_split['pos'] = torch.cat((src[index].unsqueeze(0), dst[index].unsqueeze(0)))
    edges_split['pos_type'] = type_of_edges[index]
    del src, dst, type_of_edges
    edge, edge_type = make_edge(edges_split['pos'], edges_split['pos_type'], num_nodes, args.dataset, dev)

    train_target = train_target.to(dev)
    node_features = node_features.to(args.device)
    for l in range(1):
        model = HetGDC(num_class=num_classes,
                            num_layers=num_layers,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            alpha=args.alpha,
                            type_nodes=type_nodes,
                            mode=args.mode,
                            dataset=args.dataset,
                            dev=args.device).to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss = nn.CrossEntropyLoss() 
        # Train & Valid & Test
        best_val_f1 = 0
        
        for i in range(epochs):
            model.zero_grad() 
            model.train() 
    
            neg_edges = negative_sampling(edge, num_nodes, edge.size()[1] * 30)

            z, reg_loss = model(node_features, edge, neg_edges)
            loss, _ = model.node_classification(z, train_node, train_target)
            loss += reg_loss

            print('Epoch:  ',i+1)
            print('Train - Loss: {}'.format(loss.detach().cpu().numpy()))
            loss.backward()
            optimizer.step()
            model.eval()
            
            with torch.no_grad():
                z, reg_loss = model(node_features, edge, neg_edges)
                svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(z[valid_node].cpu().detach().numpy(), valid_target.cpu().detach().numpy(), num_classes=num_classes, mode='valid')
                val_f1 = svm_micro_f1_list[0][0]
    
            if float(best_val_f1) < float(val_f1):
                best_val_f1 = val_f1
                best_z = z

                with torch.no_grad():
                    svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(z[test_node].cpu().detach().numpy(), test_target.cpu().detach().numpy(), num_classes=num_classes)
    svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(best_z[test_node].cpu().detach().numpy(), test_target.cpu().detach().numpy(), num_classes=num_classes)