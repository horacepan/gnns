import random
import numpy as np
import pdb

def one_hot(index, num_feats):
    vec = np.zeros(num_feats)
    vec[index] = 1
    return vec

class AdjGraph(object):
    '''
    This graph class stores the adj list of each vertex and a dict mapping vertex to
    a one-hot encoding of its vertex feature.
    '''
    def __init__(self, adj_matrix=None, adj_list=None, vtx_labels=None, labels_dict=None):
        '''
        Args:
            adj_matrix: numpy matrix representing the adjacency matrix of the graph
            adj_list: list of list of vertices
                IE: adj_list[i] is the list of vertices(ints) that are adjacenct for vertex i
            vtx_labels: list of strings of the labels for each vertex
                IE: vtx_labels[i] is the discrete label of vertex i
        '''
        self.vtx_labels = vtx_labels # dont really need this but keep it around for debugging
        self.labels_dict = labels_dict # ditto ^
        self.size = len(vtx_labels)
        self.vertices = list(range(self.size))
        self.edges = self._gen_edges(adj_list)
        self.one_hot_vertex_labels = {}
        self.adj_matrix = adj_matrix
        self.feature_mat = np.zeros((self.size, len(labels_dict)))

        # self.shortest_path[v, w] gives the length of the shortest path from vertex v to w.
        self.shortest_path = floyd_warshall(self)

        if adj_list:
            self.neighbors = adj_list
            self.adj_matrix = np.zeros((self.size, self.size))
            for v in range(self.size):
                self.adj_matrix[v][self.neighbors[v]]= 1
        elif adj_matrix:
            self.neighbors = [None for i in range(self.size)]
            for v in range(self.size):
                self.neighbors[v] = [i for i in range(self.size) if adj_matrix[v, i] == 1]
        else:
            raise Exception("Need to supply the adjacency list OR the entire adjacency graph!")

        # Fill the one hot vertex labels
        for v in range(self.size):
            label_index = labels_dict[vtx_labels[v]]
            self.one_hot_vertex_labels[v] = one_hot(label_index, len(labels_dict))
            self.feature_mat[v] = self.one_hot_vertex_labels[v]

    def get_label(self, v):
        '''
            v: int for the vertex number in the graph
            Returns: a one hot encoding(numpy array of length {max labels} of the label for v
        '''
        assert v < self.size
        return self.one_hot_vertex_labels[v]

    def get_neighbors(self, v):
        ''' v: int for the vertex number in the graph
            Returns: a list of the vertex neighbors of v
        '''
        assert v < self.size
        return self.neighbors[v]


    def _gen_edges(self, adj_list):
        '''
        Args:
            adj_list: list of list of vertices
                IE: adj_list[i] is the list of vertices(ints) that are adjacenct for vertex i
        Returns:
            list of (vertex(int), vertex(int)) tuples for each edge in the graph
        '''
        edges = []
        for i, nbrs in enumerate(adj_list):
            edges.extend([(i, j) for j in nbrs])

        return edges

    def __len__(self):
        return self.size

    def sub_adj(self, vertices):
        '''
        Args:
            vertices: iterable of vertices(int)
        Returns:
            numpy matrix of the sub adjacency matrix for the given vertices
        Note: The order of the vertices of this sub adjacency matrix is given by the
              order of the vertices in the original graph.
        '''
        sub_adj_mat = np.zeros((len(vertices), len(vertices)))
        sorted_vs = sorted(vertices)

        # now fill in the adj_matrix
        for i in range(len(sorted_vs)):
            for j in range(len(sorted_vs)):
                # i, j are the order, v/w are the actual vertices
                v = sorted_vs[i]
                w = sorted_vs[j]
                sub_adj_mat[i, j] = self.adj_matrix[v, w]
                sub_adj_mat[j, i] = self.adj_matrix[w, v]

        return sub_adj_mat

    def neighborhood(self, v, r):
        '''
        Args:
            v: vertex(int)
            r: the radius around v to consider
        Returns:
            list of vertices(ints) that are within a distance of r from the given vertex v
        '''
        return [w for w in self.vertices if self.shortest_path[v, w] <= r]

    def permuted(self, new_order=None, seed=0):
        '''
        Args:
            new_order: list of vertices(ints) in the desired order
                This list should contain the ints 0, 1, ..., size-1 in some order.
            seed: int for random seed
        Returns:
            A new AdjGraph object with same structure(same vertex labels and
            edges) but with a different vertex ordering(according to new_order)
        '''
        if new_order is None:
            # generate a random permutation
            random.seed(0)
            new_order = sorted(self.vertices)
            random.shuffle(new_order)
        else:
            assert len(new_order) == self.size
            assert sorted(new_order) == self.vertices

        permuted_neighbors = [None for i in range(self.size)]
        permuted_vtx_labels = {}

        # Compute the new adjacency(neighbors) list and the vertex -> labels mapping
        # according to the permutation defined by new_order
        for v in self.vertices:
            permuted_pos_v = new_order[v]
            permuted_neighbors[permuted_pos_v] = list(map(lambda z: new_order[z],
                                                      self.neighbors[v]))
            permuted_vtx_labels[permuted_pos_v] = self.vtx_labels[v]

        return AdjGraph(adj_list=permuted_neighbors, vtx_labels=permuted_vtx_labels,
                        labels_dict=self.labels_dict)


def compute_receptive_fields(graph, max_lvls):
    '''

    '''
    rfields = {lvl: {} for lvl in range(max_lvls)}
    for l in range(max_lvls):
        for v in graph.vertices:
            if l == 0:
                rfields[l][v] = set([v])
            else:
                # iterate over neighbors and union
                curr_rfield = rfields[l-1][v]
                for w in graph.neighbors[v]:
                    curr_rfield = curr_rfield.union(rfields[l-1][w])
                rfields[l][v] = curr_rfield
    return rfields

def floyd_warshall(graph):
    '''
    Use the Floyd Warshall algorithm to compute all pairs shortest paths.
    Ref: https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    Args:
        graph: an AdjGraph object
    Returns:
        numpy matrix where element (i, j) denotes the length of the shortest path
        from vertex i to vertex j
    '''
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
    print("Sub adj of vertices 0, 1, 2")
    print(g.sub_adj([0, 1, 2]))
    print("Sub adj of vertices 1, 3, 4")
    print(g.sub_adj([1, 3, 4]))
    pdb.set_trace()
