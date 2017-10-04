import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils.load_data import train_val_test_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from models.molecfingerprint import MolecFingerprintNet, MolecFingerprintNet_Adj

import argparse
import torch.optim as optim
import pdb

LOG = False

def eval_model(net, graphs, targets):
    outputs = net(graphs).data # a torch tensor
    targets = torch.Tensor(targets)
    diff = outputs - targets

    results = {}
    results['rmse'] = np.sqrt(torch.mean(diff.mul(diff)))
    results['mae'] = torch.mean(torch.abs(diff))
    return results

def logline(logfile, line):
    line = time.strftime('[%H:%M:%S] ') + line
    print(line)

    # dont write to file if log file is given
    if not LOG or logfile is None:
        return

    with open(logfile, 'a') as f:
        if line[-1] != '\n':
            line = line + '\n'
        f.write(line)

def get_logger(logfile=None):
    log = lambda x: logline(logfile, x)
    return log

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-d", dest="dataset", type=str, help="name of the dataset", default='qm9')
    parser.add_argument("-l", dest="levels", type=int, help="number of layers", default=2)
    parser.add_argument("-hi",dest="hidden", type=int, help="size of hidden layers", default=5)
    parser.add_argument("-bs", dest="batchsize", type=int, help="batch size", default=1)
    parser.add_argument("-e", dest="epochs", type=int, help="max epochs", default=10)
    parser.add_argument("-lr",dest="learning_rate", type=float, help="initial learning rate",
                        default=0.001)
    return parser.parse_args()

def make_model(hidden_size, levels):
    nfeatures = 5 #C, H, O, F, N
    #model = MolecFingerprintNet(levels, nfeatures, hidden_size, F.relu)
    model = MolecFingerprintNet_Adj(levels, nfeatures, hidden_size, F.relu)

    return model

def print_epoch_results(epoch, train_results, val_results):
    result =  "Epoch {} | train rmse: {:.3f} train mae: {:.3f}"
    result += "| val rmse: {:.3f} val mae: {:.3f}"
    result = result.format(epoch, train_results['rmse'], train_results['mae'],
                           val_results['rmse'], val_results['mae'])
    # assume the logger has been instantiated?
    log(result)


def main(picklefile, logfile=None):
    args = get_args()
    log("Starting to load data")

    # data is a dict of train/validation/test graphs, and train/val/test target values
    # keys: graphs_train, graphs_val, graphs_test, y_train, y_val, y_test
    # graphs_train/graphs_val/graphs_test: list of Graph objects
    # y_train/y_val/y_test: numpy arrays
    data = train_val_test_dataset(picklefile, train_frac=0.1,
                                  val_frac=0.1, seed=42)

    log("Done loading data")
    model = make_model(args.hidden, args.levels)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    prev_val_rmse = float('inf')
    log("Starting training...")
    for epoch in range(args.epochs):
        for i in range(0, len(data['graphs_train']), args.batchsize):
            g_batch = data['graphs_train'][i:i+args.batchsize]
            y_batch = Variable(torch.Tensor(data['y_train'][i:i+args.batchsize]))

            optimizer.zero_grad()
            outputs = model.forward(g_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0 and i > 0:
                log('   Epoch {} | Batch {} | loss: {:.3f}'.format(epoch, i, loss.data[0]))
        train_res = eval_model(model, data['graphs_train'], data['y_train'])
        val_res = eval_model(model, data['graphs_val'], data['y_val'])
        print_epoch_results(epoch, train_res, val_res)

        # Stop training if the validation error stops decreasing
        if val_res['rmse'] > prev_val_rmse and epoch > 20: # do at least 5 epochs
            break

        data['graphs_train'], data['y_train'] = shuffle(data['graphs_train'], data['y_train'])
        prev_val_rmse = val_res['rmse']

if __name__ == '__main__':
    LOGFILE = '/local/hopan/scratch/log.txt'
    #PICKLEFILE ='data/testpickle_20000.pickle'
    PICKLEFILE ='data/qm9.pickle'
    log = get_logger(LOGFILE)
    main(PICKLEFILE, LOGFILE)
