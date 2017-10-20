import numpy as np
import pdb

def one_hot(index, num_feats):
    vec = np.zeros(num_feats)
    try:
        vec[index] = 1
    except:
        pdb.set_trace()
    return vec

class AdjGraph(object):
    '''
    This graph class stores the adj list of each vertex and a dict mapping vertex to
    a one-hot encoding of its vertex feature.
    '''
    def __init__(self, adj_graph=None, adj_list=None, labels=None, labels_dict=None):
        '''
            adj: numpy matrix of the adjacency matrix
            labels: list of strings of the labels for each vertex
            labels_dict: mapping from label to integer for one hot encoding
        '''
        self.labels = labels # dont really need this but keep it around for debugging
        self.labels_dict = labels_dict # ditto ^
        self.size = len(labels)
        self.vertices = range(self.size)
        self.edges = self.get_edges(adj_list)
        self.vertex_labels = {}
        self.adj = np.zeros((self.size, self.size))
        self.feature_mat = np.zeros((self.size, len(labels_dict)))

        self.shortest_path = floyd_warshall(self)
        max_labels = len(labels_dict)

        if adj_list:
            self.neighbors = adj_list
        elif adj_graph:
            self.neighbors = [None for i in range(self.size)]
            for v in range(self.size):
                self.neighbors[v] = [i for i in range(self.size) if adj_graph[i, v] == 1]
        else:
            print("Need to supply the adjacency list OR the entire adjacency graph!")

        for v in range(self.size):
            label_index = labels_dict[labels[v]]
            self.vertex_labels[v] = one_hot(label_index, max_labels)

        for i in range(self.size):
            # self.neighbors[i] is a list of neighbors of i
            self.adj[i][self.neighbors[i]]= 1
            self.feature_mat[i] = self.vertex_labels[i]

    def get_label(self, v):
        '''
            v: int for the vertex number in the graph
            Returns: the label for v
        '''
        assert v < self.size
        return self.vertex_labels[v]

    def get_neighbors(self, v):
        '''
            v: int for the vertex number in the graph
            Returns: a list of the vertex neighbors of v
        '''
        assert v < self.size
        return self.neighbors[v]


    def get_edges(self, adj_list):
        edges = []
        for i, nbrs in enumerate(adj_list):
            edges.extend([(i, j) for j in nbrs])

        return edges

    def __len__(self):
        return self.size

    def sub_adj(self, v, radius, receptive_field):
        # get the sub adjacency matrix of the n-closest nbrs
        # adj matrix is in the order of the vertices
        sub_adj_mat = np.zeros((receptive_field, receptive_field))

        # note that neighborhood will include v
        nbrs = self.neighborhood(v, radius)
        nbrs.sort(key=lambda w: self.shortest_path[v, w])
        nbrs = nbrs[:receptive_field]

        # now fill in the adj_matrix
        for i in range(len(nbrs)):
            for j in range(len(nbrs)):
                a = nbrs[i]
                b = nbrs[j]
                sub_adj_mat[i, j] = 1 if self.shortest_path[a, b] > 0 else 0
                sub_adj_mat[j, i] = sub_adj_mat[i, j]

        return sub_adj_mat

    def neighborhood(self, v, r):
        nbrs = []
        for w in self.vertices:
            if self.shortest_path[v, w] <= r:
                nbrs.append(w)
        return nbrs

    def rfield_size(self, v, lvls):
        rfield = 0
        for w in self.vertices:
            rfield += (1 if self.shortest_path[v, w] > 0 else 0)
        return rfield

def compute_receptive_fields(graph, max_lvls):
    rfields = {}
    for l in lvls:
        for v in graph.vertices:
            if l == 0:
                rfields[l][v] = set([v])
            else:
                # iterate over neighbors and union
                curr_rfield = rfields[l-1][v]
                for w in graph.neighbors[v]:
                    curr_rfield = curr_rfield.union(rfields[l-1][w])
    return rfields

def floyd_warshall(graph):
    dist = float('inf') * np.ones((len(graph), len(graph)))
    for v in graph.vertices:
        dist[v,v] = 0

    # assume unweighted graph
    for (u, v) in graph.edges:
        dist[u, v] = 1
        dist[v, u] = 1

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

if __name__ == '__main__':
    g = np.eye(3)
    labels = ['a', 'b', 'c']
    adj_list = [[1, 2, 3], [0], [0], [0, 4], [3]]
    labels_dict = {'C': 0, 'H': 1, 'O': 2}
    labels = {0: 'C', 1: 'H', 2: 'H', 3: 'H', 4: 'O'}
    g = AdjGraph(adj_list=adj_list, labels=labels, labels_dict=labels_dict)
    print(g)
    print("Sub adj around vertex 2 radius 1 rfield 4")
    print(g.sub_adj(0, 1, 5))
    print("Sub adj around vertex 2 radius 1 rfield 2")
    print(g.sub_adj(0, 2, 5))
    pdb.set_trace()
