import numpy as np
import pdb
import time

def load_adj_matrices(fname, n=-1):
    '''
    Args:
        fname: string of the file name to read from
        n: number of graphs to load. If it's < 0, load the entire file
    Returns: list of adjacency matrices(numpy arrays)
        The file given must be of the format: 1st line = number of graphs
        Then for each graph, the file contains a line denoting the size of the graph.
        The next n lines(where n is the size of the graph) give the adjacency
        matrix of the graph.
    '''
    with open(fname, 'r') as f:
        total_graphs = int(f.readline().strip())
        if n < 0:
            n = total_graphs

        all_adj_mats = []
        while len(all_adj_mats) < n:
            graph_size = int(f.readline().strip())
            adj_mat = np.zeros((graph_size, graph_size))
            for i in range(graph_size):
                adj_mat[i, :] = map(int, f.readline().strip().split())
            all_adj_mats.append(adj_mat)
            if len(all_adj_mats) % 2000 == 0 and len(all_adj_mats) > 0:
                print("Done readings: %d / %d" %(len(all_adj_mats), n))
        return all_adj_mats


def load_adj_lists(fname, n=-1):
    '''
    Args:
        fname: string of filename to read from
        n: number of graphs to load. If it's < 0, load the entire file
    Returns:
        list of list of ints(denoting the neighbors)
        Ex: The fully connected graph C_4 will return
        [[1,2,3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    '''
    with open(fname, 'r') as f:
        total_graphs = int(f.readline().strip())
        print "n is: {}".format(n)
        if n < 0:
            n = total_graphs

        all_adj_lists = [] # each graph is rep'd by their list of adj lists
        while len(all_adj_lists) < n:
            graph_size = int(f.readline().strip())
            graph_adj_list = []
            for i in range(graph_size):
                row = map(int, f.readline().strip().split())
                i_nbrs = [j for j in range(len(row)) if row[j] > 0]
                graph_adj_list.append(i_nbrs)
            all_adj_lists.append(graph_adj_list)
            if len(all_adj_lists) % 2000 == 0 and len(all_adj_lists) > 0:
                print("Done reading: %d / %d" %(len(all_adj_lists), n))
        return all_adj_lists


def load_discrete_labels(fname, n=-1):
    '''
    Args:
        fname: string of the file to read from
        n: number of labels to load. If it's less than 0, load everything.
    Returns:
        a tuple of a list of the labels and a sorted list of the unique labels
    '''
    with open(fname, 'r') as f:
        total_labels = int(f.readline().strip())
        if n < 0:
            n = total_labels

        graph_labels = []
        unique_labels = set()

        while len(graph_labels) < n:
            labels_size = int(f.readline().strip())
            curr_label = f.readline().strip().split()
            graph_labels.append(curr_label)
            # |= is the union operator
            unique_labels |= set(curr_label)
        # make a dictionary out of it?

        labels_dict = { val: index for index, val in enumerate(sorted(unique_labels)) }
        return graph_labels, labels_dict


def load_targets(fname, n=-1, dtype=float):
    '''
    Args:
        fname: string of the file to read from
        n: number of labels to load. If it's less than 0, load everything.
    Returns:
        numpy array of the target values(floats)
    '''
    # TODO: just read n lines instead of loading everything and slicing
    with open(fname, 'r') as f:
        total_labels = int(f.readline().strip())
        if n < 0:
            n = total_labels
        assert total_labels > n
        labels = np.zeros(n)

        for i in range(n):
            labels[i] = dtype(f.readline().strip())

        return labels

if __name__ == '__main__':
    gname = '../data/qm9.graph'
    lname = '../data/qm9.atoms'
    tname = '../data/qm9.target'

    start = time.time()
    size = 3
    graphs = load_adj_matrices(gname, size)
    nbrs = load_adj_lists(gname, size)
    labels, ld = load_discrete_labels(lname, size)
    targets = load_targets(tname, size, dtype=float)
    elapsed = time.time() - start
    print("elapsed: %.2f" %elapsed)
    pdb.set_trace()
