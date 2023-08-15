import pickle
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
import numpy.random as random
from torch_geometric.utils import negative_sampling
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import to_undirected

def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score_calc(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)

def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list

def evaluate_results_nc(embeddings, labels, num_classes, mode='test'):
    if mode=='test':
        print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    if mode=='test':
        print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1])]))
        print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)

    if mode=='test':
        print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
        print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std



def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    features_0 = sp.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = sp.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = sp.load_npz(prefix + '/adjM.npz')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [features_0, features_1, features_2, features_3],\
           adjM, \
           labels,\
           train_val_test_idx

def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    features_0 = sp.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = sp.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = sp.load_npz(prefix + '/features_2.npz').toarray()

    N = features_0.shape[0] + features_1.shape[0] + features_2.shape[0]
    adj = np.zeros([N, N])
    for i in range(len(idx00)):
        adj[idx00[i][:,0], idx00[i][:,1]] = 1
        adj[idx00[i][:,1], idx00[i][:,2]] = 1
    for i in range(len(idx01)):
        adj[idx01[i][:,0], idx01[i][:,1]] = 1
        adj[idx01[i][:,1], idx01[i][:,2]] = 1

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [features_0, features_1, features_2], \
           adj, \
           labels, \
           train_val_test_idx

def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    features_0 = sp.load_npz(prefix + '/features_0.npz')
    features_1 = sp.load_npz(prefix + '/features_1.npz')
    features_2 = sp.load_npz(prefix + '/features_2.npz')
    adjM = sp.load_npz(prefix + '/adjM.npz')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [features_0, features_1, features_2],\
           adjM, \
           labels,\
           train_val_test_idx

def load_FREEBASE_data(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "data/preprocessed/FREEBASE_processed/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    N = feat_m.shape[0] + feat_d.shape[0] + feat_a.shape[0] + feat_w.shape[0]
    cnt_array = [feat_m.shape[0], feat_d.shape[0], feat_a.shape[0], feat_w.shape[0]]
    adjM = np.zeros([N,N])
    cnt = 0
    M=0
    for edges in [nei_d, nei_a, nei_w]:
        M += cnt_array[cnt]
        cnt += 1
        for i in range(len(nei_a)):
            adjM[i, M+edges[i]] = 1
            adjM[M+edges[i], i] = 1

    label = torch.FloatTensor(label)
    nei_d = [torch.LongTensor(i) for i in nei_d]
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_w = [torch.LongTensor(i) for i in nei_w]
    feat_m = torch.FloatTensor(preprocess_features(feat_m))
    feat_d = torch.FloatTensor(preprocess_features(feat_d))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_w = torch.FloatTensor(preprocess_features(feat_w))
    feat_m = torch.FloatTensor(feat_m)
    feat_d = torch.FloatTensor(feat_d)
    feat_a = torch.FloatTensor(feat_a)
    feat_w = torch.FloatTensor(feat_w)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    tmp_train = np.zeros(feat_m.shape[0])
    tmp_val = np.zeros(feat_m.shape[0])
    tmp_test = np.zeros(feat_m.shape[0])
    tmp_train[train[0]] = 1
    tmp_train[train[1]] = 1
    tmp_train[train[2]] = 1
    tmp_not_train = 1-tmp_train
    val_test = tmp_not_train.nonzero()[0]
    np.random.shuffle(val_test)

    train[0] = tmp_train.nonzero()[0]
    val[0] = val_test[:val_test.shape[0]//2]
    test[0] = val_test[:val_test.shape[0]//2]

    return [feat_m, feat_d, feat_a, feat_w], \
           adjM, \
           np.argmax(label,1), \
           {'train_idx':train[0], 'val_idx':val[0], 'test_idx':test[0]}

def make_edge(edges, edge_type, N, dataset, dev):
    num_type = edge_type.max()+1
    adj = []
    for i in range(num_type):
        row = np.array(edges[0][edge_type==i].cpu())
        col = np.array(edges[1][edge_type==i].cpu())
        data = np.array([1 for x in row])
        index = torch.LongTensor((row,col))
        v = torch.FloatTensor(data)
        tmp = torch.sparse.FloatTensor(index,v, torch.Size([N,N]))
        adj.append(tmp)

    if dataset == 'DBLP':
        paths = [[0],[1],[2],[3],[4],[5]]
    if dataset == 'IMDB':
        paths = [[0],[1],[2],[3]]
    if dataset == 'ACM':
        paths = [[0],[1],[2],[3]]
    if dataset == 'FREEBASE':
        paths = [[0],[1],[2],[3],[4],[5]]
    if dataset == 'AMINER':
        paths = [[0],[1],[2],[3]]

    for j,path in enumerate(paths):
        tmp = adj[path[-1]].to_dense()
        for i in range(len(path)-2,-1,-1):
            tmp = torch.spmm(adj[path[i]],tmp)
        
        tmp = tmp.nonzero().to(dev)
        tmp_ = torch.ones(tmp.size()[0]).long().to(dev) * j

        if j == 0:
            edge = tmp
            edge_type = tmp_

        else:
            edge = torch.cat((edge, tmp),0)
            edge_type = torch.cat((edge_type, tmp_))
        del tmp, tmp_

    return edge.t(), edge_type


def make_negative_edge(edge, neg_edge, N, neg_sample):
    src, dst = edge
    not_src, not_dst = neg_edge
    length = len(src)
    num_nodes = N

    adj = torch.ones(N,N)    
    for i,_ in enumerate(not_src):
        s = not_src[i]
        d = not_dst[i]
        for i in s:
            adj[i][d] = 0
    adj[src,dst] = 1
    return (1-adj).nonzero().t()


def preprocess_attributes(N, dataset, features_list, dev):
    total_features = 0
    for i in range(len(features_list)):
        total_features += features_list[i].shape[1]
        if dataset in ['ACM', 'IMDB']:
            continue

    node_features = torch.FloatTensor(N, total_features).to(dev)
    cnt1,cnt2 = 0,0
    for i in range(len(features_list)):
        edge_index = features_list[i].nonzero()
        if dataset == 'FREEBASE':
            edge_index = edge_index.t()
        node_features[cnt1+edge_index[0], cnt2+edge_index[1]] = torch.FloatTensor(features_list[i][edge_index[0], edge_index[1]]).to(dev)
        cnt1 += features_list[i].shape[0]
        if dataset in ['ACM', 'IMDB']:
            continue
        cnt2 += features_list[i].shape[1]  
    return node_features.to(dev)

def preprocess_edges(N, adjM, dataset, features_list, dev):
    if dataset == 'DBLP':
        num_relation = 6
    if dataset == 'IMDB':
        num_relation = 4
    if dataset == 'ACM':
        num_relation = 4
    if dataset == 'FREEBASE':
        num_relation = 6

    index = adjM.nonzero()
    src = torch.LongTensor([]).to(dev)
    dst = torch.LongTensor([]).to(dev)
    length = []
    node_types = []
    type_of_edges = torch.LongTensor([]).to(dev)
    
    cnt1 = 0
    cnt = 0
    edges = []
    not_src, not_dst = [], []

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
                
                not_src.append([i for i in range(cnt1, cnt1+features_list[i].shape[0])])
                not_dst.append([i for i in range(cnt2, cnt2+features_list[j].shape[0])])

            cnt2 += features_list[j].shape[0]
        cnt1 += features_list[i].shape[0]


    for i,edge in enumerate(edges):
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
            tmp = torch.zeros(N).to(dev)
            tmp[list(node_types[i])] = 1
            type_nodes = torch.cat((type_nodes, tmp.unsqueeze(0)))
        a = a | node_types[i]

    src, dst = src.to(dev), dst.to(dev)
    cnt = 0

    edge_type = torch.zeros(num_relation, sum(length)).to(dev)
    for i in range(num_relation):
        edge_type[i][cnt:cnt+length[i]] = 1
        cnt += length[i]
    
    edges_split = {}
    edges_split['pos'] = torch.cat((src.unsqueeze(0), dst.unsqueeze(0)))
    edges_split['pos_type'] = type_of_edges
    del src, dst, type_of_edges
    edge, _ = make_edge(edges_split['pos'], edges_split['pos_type'], N, dataset, dev)
    return edge, [not_src, not_dst], type_nodes

def divide_train_val_test_edge(edge, neg_edge):
    edge = edge[:, edge[0] < edge[1]]
    index = torch.randperm(edge.size()[1])
    train_edge = edge[:, index[:edge.size()[1]//10*6]]
    val_edge = edge[:, index[edge.size()[1]//10*1:edge.size()[1]//10*(6+1)]]
    test_edge = edge[:, index[:edge.size()[1]//10*(6+1):]]
    s,d = train_edge
    s,d = torch.cat(([s,d])), torch.cat(([d,s]))
    train_edge = torch.cat((s.unsqueeze(0), d.unsqueeze(0)))
    val_edge = to_undirected(val_edge)
    test_edge = to_undirected(test_edge)

    neg_edge = neg_edge[:, neg_edge[0] < neg_edge[1]]
    index = torch.randperm(neg_edge.size()[1])
    train_neg_edge = neg_edge[:, index[:neg_edge.size()[1]//10]]
    val_neg_edge = neg_edge[:, index[neg_edge.size()[1]//10:neg_edge.size()[1]//10 + val_edge.size()[1]//2]]
    test_neg_edge = neg_edge[:, index[neg_edge.size()[1]//10 + val_edge.size()[1]//2 : neg_edge.size()[1]//10 + val_edge.size()[1]//2 + test_edge.size()[1]//2]]
    train_neg_edge = to_undirected(train_neg_edge)
    val_neg_edge = to_undirected(val_neg_edge)
    test_neg_edge = to_undirected(test_neg_edge)
    return (train_edge, val_edge, test_edge), (train_neg_edge, val_neg_edge, test_neg_edge)
