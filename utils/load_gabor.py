import sys
import numpy as np
from itertools import chain
from readfiles import *
from datasets import GraphDataset
from six.moves import cPickle as pickle
from sklearn.utils import shuffle
from graph import AdjGraph
import time
import pdb

# DICT KEYS for the pickle file
ADJ_KEY = 'adj_lists'
LABEL_KEY = 'labels'
TARGET_KEY = 'targets'
UNIQUE_LABELS_KEY = 'unique_labels_dict'

def make_pickle(fname, obj):
   with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(fname):
    with open(fname, 'r') as f:
        return pickle.load(f)

def load_graphs_targets_pickle(picklefile):
    data = load_pickle(picklefile)
    adjs = gen_graph_list(data[ADJ_KEY], data[LABEL_KEY], data[UNIQUE_LABELS_KEY])
    return adjs, data[TARGET_KEY]

# Return a list of AdjGraph objects given a list of adj_lists and labels for the graphs.
def gen_graph_list(adj_list, labels, labels_dict):
    assert len(adj_list) == len(labels)
    return [AdjGraph(adj_list=al, labels=l, labels_dict=labels_dict)
            for (al, l) in zip(adj_list, labels)]

def subindex(arr, indices):
    if type(arr) == list:
        return [arr[i] for i in indices]
    elif type(arr) == np.ndarray:
        return arr[indices]
    else:
        print("Not sure what to do with arr of type:", type(arr))

def manual_split(*arrays, **options):
    if len(arrays) == 0:
        print("Cant do it!")
    train = options.pop('train', None)
    test = options.pop('test', None)
    iterable = ((subindex(a, train), subindex(a, test)) for a in arrays)
    return list(chain.from_iterable(iterable))

def _load_dataset(pickle_file='', graph_file='', label_file='', target_file='', n=-1):
    if '.pickle' in pickle_file:
        with open(pickle_file, 'r') as f:
            # pickle with the keys: adj_lists, labels, targets, unique_labels_dict
            pickled_dict = pickle.load(f)
            return GraphDataset(**pickled_dict)
    else:
        adj_lists = load_adj_lists(graph_file, n)
        labels, labels_dict = load_discrete_labels(label_file, n)
        targets = load_targets(target_file, n=n)
        graph_list = gen_graph_list(adj_lists, labels, labels_dict)
    return graph_list, targets

# TODO: Clean
# we don't wrap the adj graphs, labels, labels dict when pickling b/c pickle doesn't play
# nicely with classes
def make_small_pickle(sizes=None, pickle_file=''):
    if '.pickle' not in pickle_file:
        print("You need to supply a pickle file")
        return

    with open(pickle_file, 'r') as f:
        p = pickle.load(f)
        indices = range(len(p['targets']))
        [shuffled_adj, shuffled_labels, shuffled_targets] = shuffle(p['adj_lists'], p['labels'], p['targets'])
        print("Done loading %s" %pickle_file)
        for size in sizes:
            save_file = pickle_file[:-len('.pickle')] + ('%d.pickle' % size)
            small_pickle = {
                'adj_lists': shuffled_adj[:size],
                'labels': shuffled_labels[:size],
                'targets': shuffled_targets[:size],
                'unique_labels_dict': p['unique_labels_dict']
            }
            make_pickle(save_file, small_pickle)

def gen_nbrs_list(adj_graphs):
    entire_list = []
    for graph in adj_graphs:
        curr_adj_list = []
        for v in range(len(graph)):
            curr_adj_list.append([i for i in range(len(graph)) if graph[i, v] > 0])
        entire_list.append(curr_adj_list)

    return entire_list

def make_pickle_dict(adj_lists, labels, targets, unique_labels_dict, size=None):
    assert len(adj_lists) == len(labels) and len(labels) == len(targets)
    n = size if size else len(adj_lists)

    return {
        ADJ_KEY: adj_lists[:n],
        LABEL_KEY: labels[:n],
        TARGET_KEY: targets[:n],
        UNIQUE_LABELS_KEY: unique_labels_dict
    }

# TODO: Clean this
def make_pickles():
    gname = '/home/hopan/mrg/Gabor/gabor/gabor.graph'
    lname = '/home/hopan/mrg/Gabor/gabor/gabor.atoms'
    tname = '/home/hopan/mrg/Gabor/gabor/gabor.target'
    adj_graphs = load_adj_mats(gname)
    print("Done reading adj graphs")
    labels, ld = load_discrete_labels(lname)
    print("Done reading labels")
    targets = load_targets(tname)
    print("Done reading targets")
    adj_lists = gen_nbrs_list(adj_graphs)
    print("Done making adj lists")
    print(len(adj_lists))
    pickle_obj = {
        'adj_lists': adj_lists,
        'labels': labels,
        'targets': targets,
        'unique_labels_dict': ld
    }
    make_pickle('tot_energy.pickle', pickle_obj)
    print("Done pickling dictionary")
