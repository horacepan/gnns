import sys
import numpy as np
from itertools import chain
from file_io import *
from six.moves import cPickle as pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as sk_train_test_split
from graph import AdjGraph
import time
import pdb

# DICT KEYS for the pickle file
ADJ_KEY = 'adj_lists'
LABEL_KEY = 'labels'
TARGET_KEY = 'targets'
UNIQUE_LABELS_KEY = 'unique_labels_dict'
PICKLE_KEYS = [ADJ_KEY, LABEL_KEY, TARGET_KEY, UNIQUE_LABELS_KEY]

def make_pickle(fname, obj):
    '''
    Args:
        fname: string of file name to save the pickled object to
        obj: the thing to pickle
    '''
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fname):
    '''
    Args:
        fname: string of file name to unpickle
    Returns:
        the unpickled object stored in the pickle file
    '''
    with open(fname, 'r') as f:
        return pickle.load(f)


def load_graphs_targets_pickle(fname):
    '''
    Unpack the pickle and return the graphs and targets stored in the pickle file

    Args:
        fname: name of the pickle file containing a pickled dict
            The keys of the dict the entries of PICKLE_KEYS.
    Returns:
        tuple of list of graphs, and list/numpy array of target values for each graph
    '''
    data = load_pickle(fname)
    adjs = gen_graphs(data[ADJ_KEY], data[LABEL_KEY], data[UNIQUE_LABELS_KEY])
    return adjs, data[TARGET_KEY]


def gen_graphs(adj_lists, labels, labels_dict):
    '''
    Args:
        adj_lists: list of adjaceny lists for each graph
        labels: list of labels for each graph
        labels_dict: dictionary
    Returns:
        list of Graph objects
    '''
    assert len(adj_lists) == len(labels)
    return [AdjGraph(adj_list=al, labels=l, labels_dict=labels_dict)
            for (al, l) in zip(adj_lists, labels)]


def gen_nbrs_list(adj_graphs):
    '''
    Args:
        adj_graphs: list of adjacency matrices for each graph
    Returns:
        list of list of neighbors for each graph
    '''
    entire_list = []
    for graph in adj_graphs:
        curr_adj_list = []
        for v in range(len(graph)):
            curr_adj_list.append([i for i in range(len(graph)) if graph[i, v] > 0])
        entire_list.append(curr_adj_list)

    return entire_list


def subindex(arr, indices):
    '''
    Args:
        arr: list or numpy array
        indices: list of ints of indices of arr to grab
    Returns:
        a list of the values from arr at the given indices
        Ex: subindex([5, 7, 3, 99], [0, 2]) returns [5, 3]
    '''
    if type(arr) == list:
        return [arr[i] for i in indices]
    elif type(arr) == np.ndarray:
        return arr[indices]
    else:
        print("Not sure what to do with arr of type:", type(arr))


def train_test_split(*arrays, **options):
    '''
    Args:
        arrays: sequence of lists
        options: dict of
            train: list of ints(indices) to use in the train set
            test: list of ints(indices) to use in the test set
            train_size: fraction
            seed: int
            NOTE: if train_size is given, train/test_indices should not be given
    Returns:
        list containing the train test split of input arrays
    '''
    if len(arrays) == 0:
        print("Cant do it!")
        return

    train_size = options.pop('train_size', None)
    train_indices = options.pop('train', None)
    test_indices = options.pop('test', None)
    seed = options.pop('seed', 0)

    if train_size:
        return sk_train_test_split(*arrays, train_size=train_size, random_state=seed)
    else:
        iterable = ((subindex(a, train_indices), subindex(a, test_indices)) for a in arrays)
        return list(chain.from_iterable(iterable))


def train_val_test_dataset(pickle_name, train_frac, val_frac, seed=42, split=None):
    '''
    Take a pickle file that contains a dataset and return
    a dictionary of the train/validation/test graphs and targets

    Args:
        pickle_name: string of the pickle file
        train_frac: float of how much of the dataset to use for training
        val_frac: float of how much of the dataset to use for validation
        seed: int for seeding the random factor in splitting the data
    Returns:
        dictionary with the following 6 keys/values:
        g_{train/val/test}: list of Graphs for the {train/val/test} set
        y_{train/val/test}: list of target values for the {train/val/test} set
    '''
    assert 0 <= train_frac <= 1
    assert 0 <= val_frac <= 1
    assert 0 <= train_frac + val_frac <= 1

    # load the data and train test split
    graphs, targets = load_graphs_targets_pickle(pickle_name)
    print("Done loading from %s" %pickle_name)

    if split is not None:
        split = 0
        train_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/train_%d.txt' %split
        test_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/test_%d.txt' %split
        train_indices = np.loadtxt(train_split_file, skiprows=1, dtype=int)
        test_indices = np.loadtxt(test_split_file, skiprows=1, dtype=int)

        train_graphs, test_graphs, train_targets, test_targets =\
           train_test_split(graphs, targets, train=train_indices, test=test_indices)
    else:
        train_graphs, test_graphs, train_targets, test_targets = \
            train_test_split(graphs, targets, train_size=train_frac+val_frac)

    # split the training set to a train and validation set
    train_graphs, val_graphs, train_targets, val_targets = \
        train_test_split(train_graphs,
                         train_targets,
                         train_size=train_frac/(train_frac+ val_frac),
                         random_state=seed)
    print("Size of train set: %d, val set: %d, test set: %d"
        %(len(train_graphs), len(val_graphs), len(test_graphs)))

    data = {
        'graphs_train': train_graphs,
        'graphs_val': val_graphs,
        'graphs_test': test_graphs,
        'y_train': train_targets,
        'y_val': val_targets,
        'y_test': test_targets
    }
    return data


def make_dataset_pickle(gname, lname, tname, pickle_name, size=-1):
    '''
    Args:
        gname: string name of the adjacency matrices file
        lname: string name of the graph labels file
        tname: string name of the targets file
        size: int of number of graphs to store in the pickle file.
            If size < 0, use the entire dataset when loading the graph/label/target file.

    Creates a pickle file of the following format:
        {
            ADJ_KEY: list of the adjacency lists of each graphs
            LABEL_KEY: list of the labels for each vertex of each graph,
            TARGET_KEY: list of the target values for each graph,
            UNIQUE_LABELS_KEY: dictionary mapping label -> int of the labels in the dataset
        }
    '''
    adj_lists = load_adj_lists(gname, n=size)
    labels, ld = load_discrete_labels(lname, n=size)
    targets = load_targets(tname, n=size)
    assert len(adj_lists) == len(labels) and len(labels) == len(targets)

    pickle_obj = {
        ADJ_KEY: adj_lists,
        LABEL_KEY: labels,
        TARGET_KEY: targets,
        UNIQUE_LABELS_KEY: ld
    }
    make_pickle(pickle_name, pickle_obj)


if __name__ == '__main__':
    # Make a pickle file with 100 graphs
    gname = '../data/qm9.graph'
    lname = '../data/qm9.atoms'
    tname = '../data/qm9.target'
    try:
        size = sys.argv[1]
    except:
        print "Please enter the number of datapoints from the qm9 dataset to pickle."
        exit(0)
    pickle_name = '../data/testpickle_{}.pickle'.format(size)
    make_dataset_pickle(gname, lname, tname, pickle_name, size)
