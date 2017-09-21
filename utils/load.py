import pdb
import numpy as np
from load_gabor import load_graphs_targets_pickle, manual_split
from sklearn.model_selection import train_test_split

def load_dataset(dataset):
    DISPATCH_TABLE = {
        'qm9': load_qm9
    }
    if dataset in DISPATCH_TABLE:
        return DISPATCH_TABLE[dataset]()

def load_qm9(frac=None, seed=42, small=False):
    # load the data and train test split
    split = 0

    pickle_name = '/stage/risigroup/NIPS-2017/Experiments-Gabor/data/gabor.pickle'
    if small:
        # test run on the 1k pickle file
        pickle_name = '/stage/risigroup/NIPS-2017/Experiments-Gabor/data/gabor_1000.pickle'


    graphs, targets = load_graphs_targets_pickle(pickle_name)
    print("Done loading from %s" %pickle_name)

    train_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/train_%d.txt' %split
    test_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/test_%d.txt' %split
    train_indices = np.loadtxt(train_split_file, skiprows=1, dtype=int)
    test_indices = np.loadtxt(test_split_file, skiprows=1, dtype=int)

    if True:
        train_graphs, test_graphs, train_targets, test_targets =\
            train_test_split(graphs, targets, train_size=0.2)
    else:
        train_graphs, test_graphs, train_targets, test_targets =\
            manual_split(graphs, targets, train=train_indices, test=test_indices)

    train_size = frac
    train_graphs, val_graphs, train_targets, val_targets = train_test_split(train_graphs, train_targets,
                                                                            train_size=train_size,
                                                                            random_state=seed)
    print("Size of train set: %d, val set: %d, test set: %d"
        %(len(train_graphs), len(val_graphs), len(test_graphs)))

    data = {
        'g_train': train_graphs,
        'y_train': train_targets,
        'g_val': val_graphs,
        'y_val': val_targets,
        'g_test': test_graphs,
        'y_test': test_targets
    }
    return data

if __name__ == '__main__':
    data = load_qm9(0.2)
    pdb.set_trace()
