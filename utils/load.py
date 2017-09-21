import pdb
import numpy as np
from load_gabor import load_graphs_targets_pickle, manual_split
from sklearn.model_selection import train_test_split

def train_val_test_dataset(pickle_name, train_frac, val_frac, seed=42):
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

    # TODO: Disregard split files for now
    # split = 0
    #train_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/train_%d.txt' %split
    #test_split_file = '/stage/risigroup/NIPS-2017/Experiments-Gabor/splits/test_%d.txt' %split
    #train_indices = np.loadtxt(train_split_file, skiprows=1, dtype=int)
    #test_indices = np.loadtxt(test_split_file, skiprows=1, dtype=int)
    #train_graphs, test_graphs, train_targets, test_targets =\
    #   manual_split(graphs, targets, train=train_indices, test=test_indices)

    # split the data into a training/testing set
    train_graphs, test_graphs, train_targets, test_targets = \
        train_test_split(graphs, targets, train_size=train_frac+val_frac)

    # split the training set to a train and validation set
    train_graphs, val_graphs, train_targets, val_targets = \
        train_test_split(train_graphs, train_targets, train_size=train_frac/(train_frac+ val_frac),
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
