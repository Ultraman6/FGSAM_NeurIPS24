import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from layer import GraphConvolution


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, num_layers=2):
        super(GPN_Encoder, self).__init__()

        if num_layers == 1:
            gcs = [GraphConvolution(nfeat, nhid)]
        elif num_layers >= 2:
            gcs = [GraphConvolution(nfeat, 2 * nhid)]
            for _ in range(1, num_layers-1):
                gcs.append(GraphConvolution(2 * nhid, 2 * nhid))
            gcs.append(GraphConvolution(2 * nhid, nhid))
        self.gcs = nn.ModuleList(gcs)

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, adj, use_gnn=True):
        for layer in self.gcs[:-1]:
            x = F.relu(layer(x, adj, use_gnn))
            if use_gnn:
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcs[-1](x, adj, use_gnn)

        return x

    def get_params(self):
        params = []
        for layer in self.gcs:
            params.append(layer.weight)
            params.append(layer.bias)
        
        return params


class GPN_Valuator_simple(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.
    """
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator_simple, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj, use_gnn=True):
        x = F.relu(self.gc1(x, adj, use_gnn))
        if use_gnn:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj, use_gnn))
        x = self.fc3(x)

        return x


class GPN_Valuator(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.1):
        super(GPN_Valuator, self).__init__()

        self.attn1 = GATConv(nfeat, 2 * nhid, dropout=dropout)
        self.attn2 = GATConv(2 * nhid, nhid, dropout=dropout)
        self.fc = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x:torch.Tensor, adj:torch.sparse.FloatTensor, use_gnn=True):
        if use_gnn:
            # adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
            adj = adj._indices()
            x = self.attn1(x, adj).relu()
            x = F.dropout(x, self.dropout, self.training)
            x = self.attn2(x, adj)
        else:
            x = (self.attn1.lin_src(x) + self.attn1.bias).relu()
            x = self.attn2.lin_src(x) + self.attn2.bias
        x = self.fc(x)
        return x

    def get_params(self):
        params = [
            self.attn1.lin_src.weight, self.attn1.bias, 
            self.attn2.lin_src.weight, self.attn2.bias, 
        ]
        return params