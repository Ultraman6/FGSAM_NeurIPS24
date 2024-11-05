import json
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import ToUndirected
from torch_sparse import SparseTensor

from torch_geometric.datasets import (
    Amazon, 
    Coauthor, 
    CoraFull, 
    Planetoid, 
    DBLP, 
    Reddit, 
)


ds_root = "../_data"

dataset_list = ["amazon-clothing", "amazon-electronics", "dblp"]



def load_data(dataset_name:str, class_split:dict, root=None):
    if root == None:
        root = ds_root

    if dataset_name in dataset_list:
        if dataset_name == 'ogbn-arxiv':
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(root=osp.join(root, 'ogb'), name='ogbn-arxiv')
        elif dataset_name == 'corafull':
            dataset = CoraFull(root=osp.join(root, 'corafull'))
        elif dataset_name == 'coauthor-cs':
            dataset = Coauthor(root=root, name='cs')
        elif dataset_name == 'cora':
            dataset = Planetoid(root=root, name='cora')
        elif dataset_name == 'citeseer':
            dataset = Planetoid(root=root, name='citeseer')
        elif dataset_name == 'amazon-computer':
            dataset = Amazon(root=root, name='computers')
        elif dataset_name == 'reddit':
            dataset = Reddit(root=osp.join(root, 'reddit'))

        data = dataset[0]

        if dataset_name == 'ogbn-arxiv':
            data = ToUndirected()(data)

        x = data.x
        num_nodes = x.shape[0]
        dim = x.shape[1]
        y = data.y.squeeze()
        num_classes = y.unique().shape[0]
        # src, tgt = data.edge_index
        edge_index = data.edge_index

        num_class_train = class_split[dataset_name]['train']
        num_class_val = class_split[dataset_name]['val']
        num_class_test = class_split[dataset_name]['test']

        class_list_test = np.random.choice(list(range(num_classes)), num_class_test, replace=False).tolist()
        class_train_val = list(set(list(range(num_classes))).difference(set(class_list_test)))
        class_list_val = np.random.choice(class_train_val, num_class_val, replace=False).tolist()
        class_list_train = list(set(class_train_val).difference(set(class_list_val)))

    else:
        root = osp.join(root, dataset_name)
        src, tgt = [], []
        for line in open(osp.join(root, '%s_network' % (dataset_name))):
            srcl, tgtl = line.strip().split('\t')
            src.append(int(srcl))
            tgt.append(int(tgtl))

        edge_index = torch.tensor([src, tgt]).long()

        data_train = sio.loadmat(osp.join(root, '%s_train.mat' % (dataset_name)))
        data_test = sio.loadmat(osp.join(root, '%s_test.mat' % (dataset_name)))

        num_nodes = max(max(src), max(tgt)) + 1
        y = np.zeros([num_nodes, 1])
        y[data_train['Index']] = data_train['Label']
        y[data_test['Index']] = data_test['Label']
        y = y.flatten()
        y = torch.from_numpy(y).long()
        num_classes = int(y.max() + 1)

        dim = data_train['Attributes'].shape[1]
        x = np.zeros([num_nodes, dim])
        x[data_train['Index']] = data_train['Attributes'].toarray()
        x[data_test['Index']] = data_test['Attributes'].toarray()
        x = torch.from_numpy(x).float()

        # class_list_train, class_list_val, class_list_test = \
        #     json.load(open(osp.join(root, '%s_class_split.json') % (dataset_name)))

        num_class_train = class_split[dataset_name]['train']
        num_class_val = class_split[dataset_name]['val']
        num_class_test = class_split[dataset_name]['test']

        class_list_test = np.random.choice(list(range(num_classes)), num_class_test, replace=False).tolist()
        class_train_val = list(set(list(range(num_classes))).difference(set(class_list_test)))
        class_list_val = np.random.choice(class_train_val, num_class_val, replace=False).tolist()
        class_list_train = list(set(class_train_val).difference(set(class_list_val)))

    print('{}: ({}, {}), #class: {}'.format(dataset_name, num_nodes, dim, num_classes))

    class_dict_train = defaultdict(list)
    class_dict_val = defaultdict(list)
    class_dict_test = defaultdict(list)
    for i, yi in enumerate(y.tolist()):
        if yi in class_list_train:
            class_dict_train[yi].append(i)
        elif yi in class_list_val:
            class_dict_val[yi].append(i)
        else:
            class_dict_test[yi].append(i)

    return x, y, edge_index, \
        class_list_train, class_list_val, class_list_test, \
            class_dict_train, class_dict_val, class_dict_test


def task_generator(n_way, k_spt, m_qry, class_list, class_dict, num_avail=None):
    class_selected = np.random.choice(class_list, n_way, replace=False).tolist()
    idx_spt, idx_qry = [], []
    for cls in class_selected:
        if num_avail == None:
            idx_sample = np.random.choice(class_dict[cls], k_spt+m_qry, replace=False)
            idx_spt.extend(idx_sample[:k_spt])
            idx_qry.extend(idx_sample[k_spt:])
        else:
            idx_sample = np.random.choice(class_dict[cls][:num_avail], k_spt+m_qry, replace=False)
            idx_spt.extend(idx_sample[:k_spt])
            idx_qry.extend(idx_sample[k_spt:])

    # list: (N * K), (N * M)
    return idx_spt, idx_qry, class_selected


def edge_index_to_adj(edge_index:torch.Tensor, num_nodes):
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor(
        row=edge_index[1], 
        col=edge_index[0], 
        value=torch.ones([edge_index.shape[1]]).to(edge_index.device), 
        sparse_sizes=[num_nodes, num_nodes]).to_torch_sparse_coo_tensor()
    return adj


def edge_index_to_adj_with_rw_norm(edge_index:torch.Tensor, num_nodes):
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index[1], num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == torch.inf] = 0.
    norm = deg_inv[edge_index[1]]
    adj = SparseTensor(
        row=edge_index[1], 
        col=edge_index[0], 
        value=norm, 
        sparse_sizes=[num_nodes, num_nodes]).to_torch_sparse_coo_tensor()
    return adj


def edge_index_to_adj_with_gcn_norm(edge_index:torch.Tensor, num_nodes):
    # edge_index, norm = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=True)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    deg = degree(edge_index[1], num_nodes)
    deg_sqrt_inv = deg.pow(-0.5)
    deg_sqrt_inv[deg_sqrt_inv == torch.inf] = 0.
    norm = deg_sqrt_inv[edge_index[0]] * deg_sqrt_inv[edge_index[1]]
    adj = SparseTensor(
        row=edge_index[1], 
        col=edge_index[0], 
        value=norm, 
        sparse_sizes=[num_nodes, num_nodes]).to_torch_sparse_coo_tensor()
    return adj


def adj_gcn_norm(adj:torch.Tensor):
    # adj: (n, n)
    deg = adj.sum(dim=-1)
    deg_sqrt_inv = deg.pow(-0.5)
    deg_sqrt_inv[deg_sqrt_inv==torch.inf] = 0.
    deg_sqrt_inv = torch.diag(deg_sqrt_inv)
    return torch.chain_matmul(deg_sqrt_inv, adj, deg_sqrt_inv)


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def set_seed(seed=28):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)