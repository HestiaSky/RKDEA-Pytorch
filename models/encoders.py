"""Graph encoders."""
from layers.att_layers import GraphAttentionLayer
from layers.layers import *


class Encoder(nn.Module):
    # Encoder abstract class

    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class MLP(Encoder):
    # Multi-layer perceptron.

    def __init__(self, args):
        super(MLP, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class GCN(Encoder):
    # Graph Convolution Networks.

    def __init__(self, args):
        super(GCN, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    # HighWay Graph Convolution Networks.

    def __init__(self, args):
        super(HGCN, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(HighWayGraphConvolution(in_dim, out_dim, args.dropout, act, args.bias, args.cuda, args.device))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class GAT(Encoder):
    # Graph Attention Networks.

    def __init__(self, args):
        super(GAT, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


model2encoder = {
    'GCN': GCN,
    'GAT': GAT,
    'HGCN': HGCN,
    'Distill': HGCN,
}