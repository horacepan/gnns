import pdb
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import *

def chi_matrix(v_rfield, w_rfield):
    # v_rfield and w_rfield are sets/lists of vertices(ints)
    v_rfield = sorted(v_rfield)
    w_rfield = sorted(w_rfield)

    mat = torch.zeros((len(v_rfield), len(w_rfield)))

    for i in range(len(v_rfield)):
        for j in range(len(w_rfield)):
            if v_rfield[i] ==  w_rfield[j]:
                mat[i, j] = 1

    return mat

class Steerable_2D(nn.Module):
    def __init__(self, lvls, w_sizes, nonlinearity=F.relu, mode='mix'):
        '''
        Args:
            lvls: number of layers
            w_sizes: dict of the following format: {int: {'in': int, 'out': int}}
                w_sizes[L]['in'] denotes the number of in channels at lvl L
                w_sizes[L]['out'] denotes the number of out channels at lvl L
            nonlinearity: torch function for the nonlinearity to apply at each layer
            mode: string. If the mode is mix, the weight matrices will mix the channels
        '''
        super(Steerable_2D, self).__init__()
        self.mode = mode
        self.lvls = lvls
        self.w_sizes = w_sizes
        self.nonlinearity = nonlinearity
        self.init_model_variables(lvls, w_sizes)

    def forward(self, graph):
        '''
        Args:
            graph: an AdjGraph object
        Returns:
            tuple of (float prediction, Variable of the graph representation)
        '''

        '''
        num_graphs = len(graphs)
        final_channels = self.w_sizes[self.lvls-1]['out']
        graph_reprs = Variable(torch.zeros(num_graphs, final_channels))

        for i, graph in enumerate(graphs):
            for v in graph.vertices:
                vlabel = torch.Tensor(graph.get_label(v)).unsqueeze(0)

            graph_reprs[i] = self.forward_single_graph(g)

        output = self.fc_layer(graph_reprs)
        return output
        '''

        if isinstance(graph, list):
            # just do one graph while testing
            graph = graph[0]

        g_repr = self.forward_single_graph(graph)
        output = self.fc_layer(g_repr)

        return output, g_repr

    def init_model_variables(self, lvls, w_sizes):
        for lvl in range(1, self.lvls):
            if self.mode == 'mix':
                out_dim = w_sizes[lvl]['out']
                in_dim = w_sizes[lvl]['in']
                setattr(self, 'w_%d' %lvl, nn.Linear(in_dim, out_dim))
                setattr(self, 'w_%d_manual' %lvl, Variable(torch.randn(out_dim, in_dim)))
            else:
               # do the lambda I and lambda 1s instead
                setattr(self, 'lambda_eye_%d' %lvl, Variable(torch.randn(1), requires_grad=True))
                setattr(self, 'lambda_ones%d' %lvl, Variable(torch.randn(1), requires_grad=True))
                setattr(self, 'bias_%d' %lvl, Variable(torch.randn(w_sizes[lvl]['out']),
                                                     requires_grad=True))

            setattr(self, 'adj_lambda_%d' %lvl, Variable(torch.randn(1), requires_grad=True))

        self.fc_layer = nn.Linear(self.w_sizes[self.lvls-1]['out'], 1)

    def forward_single_graph(self, graph):
        '''
        Args:
            graph: Graph object
            vtx_features: dict of dicts
            IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's representation at level l

            Compute the vertex representations at each level for each vertice in the graph and
            return the graph representation of this graph
        '''

        vtx_features = self.init_base_features(graph)
        rfields = compute_receptive_fields(graph, self.lvls)

        for lvl in range(1, self.lvls):
            for v in graph.vertices:
                v_rfield = rfields[lvl][v] # receptive field of vertex v at level lvl
                v_nbrs = graph.neighborhood(v, 1)
                k = len(v_rfield)
                n = len(v_nbrs)
                in_channels = self.w_sizes[lvl]['in']
                out_channels = self.w_sizes[lvl]['out']
                aggregate = Variable(torch.zeros((n, in_channels, k, k)), requires_grad=False)
                reduced_adj_mat = Variable(torch.Tensor(graph.sub_adj(v_rfield)), requires_grad=False)

                for index, w in enumerate(v_nbrs):
                    w_rfield_prev = rfields[lvl-1][w] # receptive field of vertex w
                    chi = Variable(chi_matrix(v_rfield, w_rfield_prev), requires_grad=False)
                    nbr_feat = vtx_features[lvl-1][w] # should be a Variable already
                    aggregate[index] = aggregate[index].add(chi.matmul(nbr_feat).matmul(chi.t()))

                aggregate = aggregate.sum(dim=0) # collapse on the neighbors
                aggregate = aggregate.add(self.adj_param(lvl) * reduced_adj_mat)
                # Before reshaping aggregate has shape (k, k, in_channels), where k is the size
                # of the receptive field of vertex v.

                # After mixing channels via the w matrix, we get a tensor
                # of shape (out_channels, k*k)
                new_features = self.nonlinearity(self.linear_transform(lvl, aggregate.view(k*k, -1)))

                # new features will be of size k*k, new_channels
                # Reshape it to be of size (k, k, out_channels)
                vtx_features[lvl][v] = new_features.view(out_channels, k, k)

        graph_repr = self.collapse_vtx_features(vtx_features)
        return graph_repr

    def init_base_features(self, graph):
        '''
        Fills vtx_features with the level 0 representations of the vertices
        of the given graph

        Args:
            graph: AdjGraph object
            vtx_features: dict to fill with vtx features
                IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's
                    representation at level l
        '''
        vtx_features = { lvl: {} for lvl in range(self.lvls) }

        for v in graph.vertices:
            # vlabel is a numpy array
            #vlabel = graph.get_scalar_label(v)
            vlabel = graph.get_label(v)
            if isinstance(vlabel, (int, float)):
                vlabel = torch.Tensor([vlabel]).view(1,1,1)
            else:
                vlabel = torch.Tensor(vlabel.reshape((len(vlabel), 1, 1)))

            vtx_features[0][v] = Variable(vlabel, requires_grad=False)

        return vtx_features

    def w(self, lvl):
        '''
        Returns:
            The linear layer at level lvl
        '''
        return getattr(self, 'w_%d' %lvl)

    def w_manual(self, lvl):
        '''
        Returns:
            The weight matrix for the linear layer at level lvl
        '''

        return getattr(self, 'w_%d_manual' %lvl)

    def linear_transform(self, lvl, input):
        '''
        Apply the appropriate linear transform according to the "mode" of
        this network.
        '''
        if self.mode == 'mix':
            #return F.linear(input, self.w_manual(lvl))
            return self.w(lvl)(input)
        else:
            pass



    def adj_param(self, lvl):
        '''
        Returns the learnable parameter that scales the adjacency matrix at level lvl
        '''
        assert 0 <= lvl < self.lvls
        return getattr(self, 'adj_lambda_%d' %lvl)

    def collapse_vtx_features(self, vtx_features):
        '''
        Args:
            vtx_features: a dict of dicts
            IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's representation at level l
        Collapse the final level vertex features to nchannels(num channelsof final layer)
        Sum these from all vertices in the graph
        '''
        graph_repr = Variable(torch.zeros(1, self.w_sizes[self.lvls-1]['out']),
                              requires_grad=True)
        for v in vtx_features:
            # collapse it so that it's dimension is just the number of channels
            graph_repr = graph_repr.add(vtx_features[self.lvls-1][v].sum(-1).sum(-1))

        return graph_repr

def test_permutation_invariance(_graph=None):
    '''
    Creates a simple graph
    '''
    torch.manual_seed(seed=0)
    if _graph is None:
        adj_list = [[1, 2, 3], [0], [0], [0, 4], [3]]
        labels_dict = {'C': 0, 'H': 1, 'O': 2}
        vtx_labels = {0: 'C', 1: 'H', 2: 'H', 3: 'H', 4: 'O'}
        _graph = AdjGraph(adj_list=adj_list, vtx_labels=vtx_labels, labels_dict=labels_dict)

    permuted_graph = _graph.permuted(seed=0)

    # model params
    lvls = 2
    out_channel_size = 5
    w_sizes = { 0: {'out':3} }
    for lvl in range(1, lvls):
        w_sizes[lvl] = {}
        w_sizes[lvl]['in'] = w_sizes[lvl-1]['out']
        w_sizes[lvl]['out'] = out_channel_size

    # feed both graphs through the network
    model = Steerable_2D(lvls, w_sizes)
    _, graph_repr = model(_graph)
    print('=' * 80)
    _, permuted_graph_repr = model(permuted_graph)

    print("Graph representation of original graph:")
    print(graph_repr.data)
    print("Graph representation of permuted graph:")
    print(permuted_graph_repr.data)
    assert graph_repr.data.equal(permuted_graph_repr.data), "Graph reprs are not equal!"

if __name__ == '__main__':
    test_permutation_invariance()
