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

ADJ_INDEX = 0
LABEL_INDEX = 1
LABEL_DICT_INDEX = 2
TARGET_INDEX = 3

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
    adjs = gen_graph_list(data['adj_lists'], data['labels'], data['unique_labels_dict'])
    return adjs, data['targets']

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

# Return a list of AdjGraph objects given a list of adj_lists and labels for the graphs.
def gen_graph_list(adj_list, labels, labels_dict):
    assert len(adj_list) == len(labels)
    return [AdjGraph(adj_list=al, labels=l, labels_dict=labels_dict)
            for (al, l) in zip(adj_list, labels)]

def load_dataset(pickle_file='', graph_file='', label_file='', target_file='', n=-1):
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
    if size:
        n = size
    else:
        n = len(adj_lists)
    return {
        'adj_lists': adj_lists[:n],
        'labels': labels[:n],
        'targets': targets[:n],
        'unique_labels_dict': unique_labels_dict
    }

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

if __name__ == '__main__':
    dataset = sys.argv[1]
    gname_gabor = '/home/hopan/mrg/Gabor/gabor/gabor.graph'
    gname_hcep = '/stage/risigroup/NIPS-2017/Experiments-HCEP/data/hcep.nips.graph'
    gname_nci = '/stage/risigroup/NIPS-2017/Experiments-NCI/data/NCI.graph'
    lname_gabor = '/home/hopan/mrg/Gabor/gabor/gabor.atoms'
    lname_hcep = '/stage/risigroup/NIPS-2017/Experiments-HCEP/data/hcep.nips.atoms'
    lname_nci = '/stage/risigroup/NIPS-2017/Experiments-NCI/data/NCI.atoms'
    tname_gabor = '/home/hopan/mrg/Gabor/gabor/gabor.target'
    tname_hcep = '/stage/risigroup/NIPS-2017/Experiments-HCEP/data/hcep.nips.pce'
    tname_nci = '/stage/risigroup/NIPS-2017/Experiments-NCI/data/NCI.target'

    if dataset == 'Gabor':
        gname = gname_gabor
        lname = lname_gabor
        tname = tname_gabor
    elif dataset == 'HCEP':
        gname = gname_hcep
        lname = lname_hcep
        tname = tname_hcep

    elif dataset == 'NCI':
        gname = gname_nci
        lname = lname_nci
        tname = tname_nci
    else:

        print("Not a valid dataset")
    #graph_list, t = load_dataset(graph_file=gname, label_file=lname, target_file=tname, n=3)
    start = time.time()
    adj_lists = load_adj_lists(gname, n=10000)
    if dataset == 'NCI':
        labels, ld = load_nci_labels(lname, n=10000)
    else:
        labels, ld = load_discrete_labels(lname, n=10000)
    targets = load_targets(tname, n=10000)
    #load_graphs_targets_pickle('hcep.pickle')
    elapsed = time.time() - start
    print("Load time: %.2f" %elapsed)


    for size in [1000, 10000]:
        pickle_dict_small = make_pickle_dict(adj_lists, labels, targets, ld, size=size)
        pickle_fname = "/stage/risigroup/NIPS-2017/Experiments-%s/data/%s_%d.pickle" %(dataset,dataset.lower(), size)
        make_pickle(pickle_fname, pickle_dict_small)
