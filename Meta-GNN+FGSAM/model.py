import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, act=F.relu):
        super(GraphConv, self).__init__()

        self.w = Parameter(torch.zeros([in_channels, out_channels]))
        self.b = Parameter(torch.zeros(out_channels))

        self.dropout = dropout
        self.act = act

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.w)

    def forward(self, x, adj, w_b=None):
        if adj is not None:
            x = F.dropout(x, self.dropout, self.training)
        if w_b == None:
            w = self.w
            b = self.b
        else:
            w, b = w_b
        x = torch.mm(x, w)
        if adj is not None:
            x = torch.spmm(adj, x)
        x = x + b
        x = self.act(x)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.gc1 = GraphConv(in_channels, hidden_channels, dropout=0., act=F.relu)
        self.gc2 = GraphConv(hidden_channels, out_channels, dropout=dropout, act=lambda x:x)

    def forward(self, x, adj, gc_w_b=None):
        if gc_w_b == None:
            gc1_w_b = gc2_w_b = None
        else:
            gc1_w_b = gc_w_b[:2]
            gc2_w_b = gc_w_b[2:]
        x = self.gc1(x, adj, gc1_w_b)
        x = self.gc2(x, adj, gc2_w_b)
        return x


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()

        self.w = Parameter(torch.zeros([in_dim, out_dim]))
        self.b = Parameter(torch.zeros([out_dim]))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.w)

    def forward(self, x, fc_w_b=None):
        if fc_w_b is None:
            return torch.mm(x, self.w) + self.b
        else:
            w, b = fc_w_b
            return torch.mm(x, w) + b


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x, fc_w_b=None):
        if fc_w_b == None:
            return self.fc2(self.fc1(x).relu())
        else:
            fc1_w, fc1_b, fc2_w, fc2_b = fc_w_b
            out = self.fc1(x, [fc1_w, fc1_b]).relu()
            return self.fc2(out, [fc2_w, fc2_b])