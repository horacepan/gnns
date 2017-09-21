from graph import AdjGraph
from torch.utils.data import Dataset
from sklearn.utils import shuffle

import pdb
# We subclass Dataset so that we can wrap it in torch.utils.data.Dataloader
# for training which does the minibatch magic for us.
class GraphDataset(Dataset):
    def __init__(self, adj_lists=None, labels=None, unique_labels_dict=None,
                 adjgraphs=None, targets=None):
        # Either adj + labels + labels_dict + target are supplied
        # OR: adjgraphs and targets are supplied
        self.targets = targets
        self.labels = labels
        self.unique_labels_dict = unique_labels_dict
        if adjgraphs:
            self.graphs = adjgraphs
        else:
            self.graphs = [ AdjGraph(adj_list=a, labels=l, labels_dict=unique_labels_dict)
                            for a, l in zip(adj_lists, labels) ]

    def __getitem__(self, index):
        return self.graphs[index], self.targets[index]

    def __len__(self):
        return len(self.graphs)

    def subsample(self, size):
        shuffled_graphs, shuffled_targets = shuffle(self.graphs), shuffle(self.targets)
        sampled_graphs, sampled_targets = shuffled_graphs[:size], shuffled_targets[:size]
        return GraphDataset(adjgraphs=sampled_graphs, targets=sampled_targets)

