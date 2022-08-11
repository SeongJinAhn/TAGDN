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

def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
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
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
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


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "./data/freebase/"
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


def make_negative_edge(src, dst, N, neg_sample):
    length = len(src)
    num_nodes = N

    ########################
    N,M=0,0
    node2index_s = {}
    index2node_s = {}
    node2index_d = {}
    index2node_d = {}

    index2node_s_list = []
    index2node_d_list = []

    for i in src.tolist():
        if i not in index2node_s:
            node2index_s[len(node2index_s)] = i
            index2node_s[i] = len(index2node_s)
            N+=1
        index2node_s_list.append(index2node_s[i])

    index2node_s_tensor = torch.zeros(torch.max(src)+1).cuda(0)
    for i in src.tolist():
        index2node_s_tensor[index2node_s[i]] = i

    for i in dst.tolist():
        if i not in index2node_d:
            node2index_d[len(node2index_d)] = i
            index2node_d[i] = len(index2node_d)
            M+=1
        index2node_d_list.append(index2node_d[i])

    index2node_d_tensor = torch.zeros(torch.max(dst)+1).cuda(0)
    for i in dst.tolist():
        index2node_d_tensor[index2node_d[i]] = i


    src_ = []
    dst_ = []
    src = index2node_s_list
    dst = index2node_d_list
    ########################

    data = np.ones(length)
    src, dst = np.array(src), np.array(dst)

    A = sp.csc_matrix((data, (src,dst)), shape=(N,M)).todense()
    tmp = (1-A).nonzero()
    tmp = torch.cat((torch.LongTensor(tmp[0]).unsqueeze(0),torch.LongTensor(tmp[1]).unsqueeze(0)),0)#.cuda()

    np.random.seed(np.random.randint(1,100))
    if len(tmp[0]) == 0:
        return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
    index = np.random.choice(len(tmp[0]), neg_sample * len(src))

    index = torch.LongTensor(index)
    node2index_s = index2node_s_tensor[tmp[0][index]].long()
    node2index_d = index2node_d_tensor[tmp[1][index]].long()

    del A, tmp, src_, dst_, index2node_s_list, index2node_d_list, index
    return node2index_s, node2index_d