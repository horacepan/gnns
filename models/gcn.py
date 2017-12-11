'''
Graph Convolutional Networks(Kipf, Welling 2017)
https://arxiv.org/abs/1609.02907
https://tkipf.github.io/graph-convolutional-networks/
https://github.com/tkipf/pygcn
'''
import torch.functional as F
import torch.nn as nn
from torch.autograd import Parameter, Variable

def GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

    def forward(self, input, adj):
        '''
        input(torch.Tensor):
        adj(torch.Tensor): adjacency matrix
        '''
        hidden = torch.mm(input, weight)
        output = torch.mm(adj, hidden)
        if self.bias:
            return self.bias + output
        else:
            return output

class GraphConvNetwork(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        self._lvls = 3
        self.gc1 = GraphConvLayer(in_feats, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.gc3 = GraphConvLayer(hidden_size, out_feats)

    def forward(self, feats, adj):
        '''
        feats(torch.tensor): the graph node features(matrix of Nxd), N = num nodes, d = num features
        adj(torch.tensor): adjacency matrix of the graph

        '''
        x = F.relu(self.gc1(feats, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        return x
