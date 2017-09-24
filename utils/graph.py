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
        self.vertex_labels = {}
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
            #else:
            #    rand_orth_features = {}
            #    for label in labels_dict.key():
            #        rand_orth_features[label] = np.random.normal(max_labels)
            #        rand_orth_features[label] /= np.linalg.norm(rand_orth_features[label])

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

if __name__ == '__main__':
    g = np.eye(3)
    labels = ['a', 'b', 'c']
