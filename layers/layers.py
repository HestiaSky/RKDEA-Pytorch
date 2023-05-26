import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


def get_dim_act(args):
    # Helper function to get dimension and activation at every layer.
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers-1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers-1))
    return dims, acts


class GraphConvolution(Module):
    # Simple GCN layer.

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(self.in_features, self.out_features)


class HighWayGraphConvolution(GraphConvolution):
    # GCN Layer with HighWay Gate

    def __init__(self, in_features, out_features, dropout, act, use_bias, cuda, device):
        super(HighWayGraphConvolution, self).__init__(in_features, out_features, dropout, act, use_bias)
        assert (self.in_features == self.out_features)
        d = self.in_features
        init_range = np.sqrt(6.0 / (d + d))
        self.kernel_gate = torch.FloatTensor(d, d).uniform_(-init_range, init_range)
        self.bias_gate = torch.zeros([d])
        if not cuda == -1:
            self.kernel_gate = self.kernel_gate.to(device)
            self.bias_gate = self.bias_gate.to(device)

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        support = self.act(support)

        transform_gate = torch.spmm(x, self.kernel_gate) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        if x.is_sparse:
            residual = x.to_dense()
        else:
            residual = x
        output = transform_gate * support + carry_gate * residual, adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(self.in_features, self.out_features)


class Linear(Module):
    # Simple Linear layer with dropout.

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out

