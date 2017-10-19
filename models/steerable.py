import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Steerable_2D(nn.module):
    def __init__(self, lvls, channels, receptive_field, nchannels, nonlinearity=F.sigmoid):
        '''
        Args:
            lvls: number of layers
            max_k: max size of receptive field
            channesl: number of channels(lenght of vertex labels)
        '''

        self.lvls = lvls
        self.channels = channels
        self.receptive_field = receptive_field
        self.nchannels = nchannels
        self.nonlinearity = nonlinearity
        self.fc_layer = nn.Linear()
        self.model_variables = {}
        self.init_model_variables(lvls)

    def forward(self, graphs):
        '''
        Args:
            graphs: list of Graph objects
        '''

        num_graphs = len(graphs)
        rfields = compute_receiptive_fields(graph, self.lvls)
        vtx_features = {}
        final_channels = self.get_nchannels(self.lvls) # num channels at final layer
        graph_reprs = Variable(torch.zeros(num_graphs, final_channels))

        for i, graph in enumerate(graphs):
            vtx_features = {}
            self.init_base_features(graph, vtx_features)
            self.forward_single_graph(g, vtx_features)
            g_repr = self.collapse_vtx_features(vtx_features)
            graph_reprs[i].add(g_repr)

        output = self.fc_layer(graph_reprs)
        return output

    def init_model_variables(self, lvls):
        # TODO: fill in the dimensions
        self.fc_layer = nn.Linear()
        for l in self.lvls:
            # TODO: dont use strings
            setattr(self, 'bias_%d' %l, Variable())
            setattr(self, 'w_%d' %l, Variable())
            setattr(self, 'bias_%d' %l, Variable())
            setattr(self, 'lambda_eye_%d' %l, Variable())
            setattr(self, 'lambda_ones_%d' %l, Variable())
            setattr(self, 'lambda_adj_%d' %l, Variable())

    def forward_single_graph(self, graph, vtx_features):
        '''
        Args:
            graph: Graph object
            vtx_features: dict of dicts
            IE: vtx_features[l][v] gives a torch.Tensor of the vertex v's representation at level l

            Compute the vertex representations at each level for each vertice in the graph
        '''
        vtx_features = {l: {} for l in range(self.lvls+1)}
        rfields = compute_receiptive_fields(graph, self.lvls)

        for lvl in range(1, self.levels + 1):
            for v in graph.vertices
                reduced_adj_mat = Variable(graph.sub_adj(v, lvl), requires_grad=False)
                nchannels = self.get_nchannels(lvl)
                k = graph.rfield_size(v, lvl)
                aggregate = Variable(torch.zeros((k, k, k, nchannels), requires_grad=True))

                # TODO: implement get_receptive_field, chi_matrix(rename maybe) in graph.py
                for index, w in enumerate(graph.neighborhood(v, lvl)):
                    chi_matrix = Variable(graph.chi_matrix(v, w, lvl), requires_grad=False)
                    nbr_feat = vtx_features[lvl-1][w] # should be a Variable already
                    aggregate[index] = agregate[index].add(chi_matrix.matmul(nbr_feat).matmul(chi_matrix.t()))

                collapsed = aggregate.sum(dim=0) # collapse on the neighbors

                aggregate = aggregate.add(self.adj_param(lvl) * reduced_adj_mat)
                vtx_features[lvl][v] = self.nonlinearity(self.apply_w_bias(aggregate, lvl))

        return vtx_features

    def self.apply_w_bias(self, vtx_repr, W, bias):
        pass

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


    def get_nchannels(self, lvl):
        return 5
        #return self.nchannels * (2 ** lvl)

    def rfield_size(self, lvl):
        return lvl * self.receptive_field

    def init_base_features(self, vtx_features):
        pass

    def backward(self):
        pass
