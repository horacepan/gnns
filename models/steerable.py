import pdb
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def chi_matrix(v_rfield, w_rfield):
    # v_rfield and w_rfield are lists of vertices(ints)
    v_rfield.sort()
    w_rfield.sort()

    mat = np.zeros((len(v_rfield), len(w_rfield)))

    for i in range(len(v_rfield)):
        for j in range(len(w_rfield)):
            if v_rfield[i] ==  w_rfield[j]:
                mat[i, j] = 1
    return mat

class Steerable_2D(nn.Module):
    def __init__(self, lvls, channels, w_sizes, nonlinearity=F.sigmoid, mode='mix'):
        '''
        Args:
            lvls: number of layers
            max_k: max size of receptive field
            channesl: number of channels(lenght of vertex labels)
        '''

        self.lvls = lvls
        self.channels = channels
        self.nonlinearity = nonlinearity
        self.fc_layer = nn.Linear()
        self.model_variables = {}
        self.init_model_variables(lvls, w_sizes)

    def forward(self, graphs):
        '''
        Args:
            graphs: list of Graph objects
        '''

        num_graphs = len(graphs)
        vtx_features = {}
        final_channels = self.w_sizes[self.lvls]['out']
        graph_reprs = Variable(torch.zeros(num_graphs, final_channels))

        for i, graph in enumerate(graphs):
            vtx_features = {}
            for v in graph.vertices:
                vlabel = torch.Tensor(graph.get_label(v)).unsqueeze(0)
                vtx_features[0][v] = Variable(vlabel, requires_grad=False).unsqueeze(0)

            self.forward_single_graph(g, vtx_features)
            g_repr = self.collapse_vtx_features(vtx_features) # take only top level?
            graph_reprs[i] = graph_repres[i].add(g_repr)

        output = self.fc_layer(graph_reprs)
        return output

    def init_model_variables(self, lvls, w_sizes):
        for l in range(self.lvls):
            if self.mode == 'mix':
                setattr(self, 'w_%d' %l, nn.Linear(w_sizes[lvl]['in'], w_sizes[lvl]['out']))
            else:
               # do the lambda I and lambda 1s instead
                setattr(self, 'lambda_eye_%d' %l, Variable(torch.randn(), requires_grad=True))
                setattr(self, 'lambda_ones%d' %l, Variable(torch.randn(), requires_grad=True))
                setattr(self, 'bias_%d' %l, Variable(torch.randn(w_sizes[lvl]['out']),
                                                     requires_grad=True))

            setattr(self, 'adj_lambda_%d' %l, Variable(torch.randn(), requires_grad=True))

        self.fc_layer = nn.Linear(self.w_sizes[self.lvls]['out'], 1)

    def forward_single_graph(self, graph, vtx_features):
        '''
        Args:
            graph: Graph object
            vtx_features: dict of dicts
            IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's representation at level l

            Compute the vertex representations at each level for each vertice in the graph
        '''
        vtx_features = {l: {} for l in range(self.lvls+1)}
        rfields = compute_receptive_fields(graph, self.lvls)

        for lvl in range(1, self.levels + 1):
            for v in graph.vertices:
                reduced_adj_mat = Variable(graph.sub_adj(v, lvl), requires_grad=False)
                k = graph.rfield_size(v, lvl)
                n = len(graph.neighborhood(v, lvl))
                in_channels = self.w_sizes[lvl]['in']
                out_channels = self.w_sizes[lvl]['out']
                aggregate = Variable(torch.zeros((n, k, k, in_channels, requires_grad=False)))
                v_rfield = rfields[lvl][v] # receptive field of vertex v

                for index, w in enumerate(graph.neighborhood(v, lvl)):
                    w_rfield_prev = rfields[lvl][w] # receptive field of vertex w
                    chi = Variable(chi_matrix(v_rfield, w_rfield_prev), requires_grad=False)
                    nbr_feat = vtx_features[lvl-1][w] # should be a Variable already
                    aggregate[index] = agregate[index].add(chi.matmul(nbr_feat).matmul(chi.t()))

                aggregate = aggregate.sum(dim=0) # collapse on the neighbors
                aggregate = aggregate.add(self.adj_param(lvl) * reduced_adj_mat)
                # Before reshaping aggregate has shape (k, k, in_channels), where k is the size
                # of the receptive field of vertex v.

                # After mixing channels via the w matrix, we get a tensor
                # of shape (out_channels, k*k)
                new_features = self.nonlinearity(self.w(lvl)(aggregate.view(in_channels, -1)))

                # Reshape it to be of size (k, k, out_channels)
                vtx_features[lvl][v] = new_features.view(k, k, out_channels)

        return vtx_features

    def w(self, lvl):
        return getattr(self, 'w_%d' %l)

    def adj_param(self, lvl):
        assert 0 <= lvl <= self.lvls
        return getattr(self, 'adj_%d' %lvl)

    def collapse_vtx_features(self, vtx_features):
        '''
        Args:
            vtx_features: a dict of dicts
            IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's representation at level l
        Collapse the final level vertex features to nchannels(num channelsof final layer)
        Sum these from all vertices in the graph
        '''
        graph_repr = Variable(torch.zeros(1, 1, self.w_sizes[lvl]['out']),
                              requires_grad=True)
        for v in vtx_features:
            # collapse it so that it's dimension is just the number of channels
            graph_repr.add(vtx_features[lvl][v].sum(0).sum(0))

        return graph_repr

    def init_base_features(self, graph, vtx_features):
        vlabel = torch.Tensor(graph.get_label(v)).unsqueeze(0)
        vertex_features = { v: Variable(vlabel, requires_grad=False).unsqueeze(0)
                            for v in graph.vertices }

