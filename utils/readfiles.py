import numpy as np
import pdb
import time
def load_adj_matrices(fname, n=-1):
    '''
    fname: string of the file name to read from
    n: number of graphs to load. If it's -1, load the entire file
    The file given must be of the format:
    1st line = number of graphs
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
                print("Done reading: %d / %d" %(len(all_adj_mats), n))
        return all_adj_mats

def load_adj_lists(fname, n=-1):
    with open(fname, 'r') as f:
        total_graphs = int(f.readline().strip())
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
    fname: string of the file to read from
    n: number of labels to load. If it's less than 0, load everything.
    Returns: a tuple of a list of the labels and a sorted list of the unique labels
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

def load_nci_labels(fname, n=-1):
    '''
    fname: string of the file to read from
    n: number of labels to load. If it's less than 0, load everything.
    Returns: a tuple of a list of the labels and a sorted list of the unique labels
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
            curr_label = map(lambda x: x if '.' not in x else x[:x.index('.')], curr_label)
            graph_labels.append(curr_label)
            # |= is the union operator
            unique_labels |= set(curr_label)
        # make a dictionary out of it?

        labels_dict = { val: index for index, val in enumerate(sorted(unique_labels)) }
        return graph_labels, labels_dict

def load_vec_labels(fname, n=-1):
    with open(fname, 'r') as f:
        total_labels = int(f.readline().strip())
        if n < 0:
            n = total_labels

        labels = []
        # TODO: figure out the interface for this
        '''
        while len(labels) < n:
            labels_size = int(f.readline().strip())
            labels.append(f.readline().strip().split())
        '''
        return labels

def load_targets(fname, skiprows=1, n=-1):
    # TODO: just read n lines instead of loading everything and slicing
    all_targets = np.loadtxt(fname, skiprows=skiprows)
    if n < 0:
        return all_targets
    print('read targets n: ', n)
    return all_targets[:n]

if __name__ == '__main__':
    gname = '/home/hopan/mrg/Gabor/gabor/gabor.graph'
    lname = '/home/hopan/mrg/Gabor/gabor/gabor.atoms'
    tname = '/home/hopan/mrg/Gabor/gabor/gabor.target'
    lname_nci = '/stage/risigroup/NIPS-2017/Experiments-NCI/data/NCI.atoms'
    #graphs = load_adj_matrices(gname, 3)
    #nbrs = load_adj_lists(gname, 3)
    start =time.time()
    labels, ld = load_nci_labels(lname_nci)
    elapsed=time.time() - start
    print("elapsed: %.2f" %elapsed)
    #targets = load_targets(tname, 3)
    pdb.set_trace()
