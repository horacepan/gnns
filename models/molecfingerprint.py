import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

def gen_w_sizes(input_features, hidden_size, levels):
    w_sizes = {i: {'in': hidden_size, 'out': hidden_size} for i in range(1, levels + 1)}
    w_sizes[1]['in'] = input_features
    return w_sizes

def gen_h_sizes(input_features, hidden_size, levels):
    h_sizes = {i: {'in': hidden_size, 'out': hidden_size} for i in range(1, levels + 1)}
    return h_sizes


# This network implements a a linear regressor on top of the molecular fingerprints
# Reference: "Convolutional Networks on Graphs for Learning Molecular Fingerprints"
# http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf
class MolecFingerprintNet(nn.Module):
    # The paper calls it "radius" but we use levels for consistency across
    # graph convolution networks.
    def __init__(self, levels, nfeatures, hidden_size, activation_func, mode='regression'):
        super(MolecFingerprintNet, self).__init__()
        self.w_sizes = gen_w_sizes(nfeatures, hidden_size, levels)
        self.h_sizes = gen_h_sizes(nfeatures, hidden_size, levels)
        self.levels = levels
        self.nfeatures = nfeatures
        self.activation_func = activation_func
        self.mode = mode

        if mode == 'regression':
            self.fc_output = nn.Linear(self.w_sizes[levels]['out'], 1)
        elif mode == 'classification':
            self.fc_output = nn.Linear(self.w_sizes[levels]['out'], 2)


        for lvl in range(1, levels+1):
            # See page 3 of algorithm 2 in
            # https://arxiv.org/abs/1509.09292
            # Hidden weights
            setattr(self, 'H_%d' %lvl, nn.Linear(self.w_sizes[lvl]['in'], self.w_sizes[lvl]['out']))
            # Output weights
            setattr(self, 'W_%d' %lvl, nn.Linear(self.h_sizes[lvl]['in'], self.h_sizes[lvl]['out']))

    def forward(self, graphs):
        num_graphs = len(graphs)
        fingerprints = Variable(torch.zeros([num_graphs, self.h_sizes[1]['out']]))
        for i, graph in enumerate(graphs):
            vertex_features = {v: Variable(torch.Tensor(graph.get_label(v)), requires_grad=False).unsqueeze(0)
                               for v in graph.vertices}
            curr_lvl_features = {}
            for lvl in range(1, self.levels + 1):
                for v in graph.vertices:
                    aggregate = vertex_features[v]
                    for neighbor in graph.get_neighbors(v):
                        # sum the stuff at previous level
                        aggregate.add(vertex_features[neighbor]) # line 8
                    hidden_activation = self.get_h(lvl)(aggregate) # line 9
                    new_vtx_feature = self.activation_func(hidden_activation) # line 9
                    sparse_output = F.softmax(self.get_w(lvl)(new_vtx_feature))# line 10
                    fingerprints[i] = fingerprints[i] + sparse_output  # line 11
                    curr_lvl_features[v] = new_vtx_feature
                vertex_features = curr_lvl_features

        output = self.fc_output(fingerprints)
        if self.mode == 'classification':
            output = nn.LogSoftmax(output)

        return output

    def get_w(self, lvl):
        assert lvl <= self.levels
        return getattr(self, 'W_%d' %lvl)

    def get_h(self, lvl):
        assert lvl <= self.levels
        return getattr(self, 'H_%d' %lvl)

class MolecFingerprintNet_Adj(MolecFingerprintNet):
    def forward(self, graphs):
        num_graphs = len(graphs)
        fingerprints = Variable(torch.zeros([num_graphs, self.h_sizes[1]['out']]))

        for i, graph in enumerate(graphs):
            graph_feat_tensor = Variable(torch.Tensor(graph.feature_mat), requires_grad=False)
            adj_mat = Variable(torch.Tensor(graph.adj_matrix), requires_grad=False)
            curr_lvl_features = {}
            for lvl in range(1, self.levels + 1):
                # sum neighboring vertices features
                aggregate = adj_mat.matmul(graph_feat_tensor)
                hidden_activation = self.get_h(lvl)(aggregate)
                new_vtx_feature = self.activation_func(hidden_activation)
                sparse_output = F.softmax(self.get_w(lvl)(new_vtx_feature))
                fingerprints[i] = sparse_output.sum(dim=0)
                graph_feat_tensor = new_vtx_feature

        output = self.fc_output(fingerprints)
        if self.mode == 'classification':
            output = nn.LogSoftmax(output)

        return output
