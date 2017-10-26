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

def eval_classification_model(net, graphs, targets, criterion):
    # output of net(graph) is softmax thing
    results = {}
    outputs = net(graphs)
    var_targets = Variable(torch.LongTensor(targets.astype(int)))

    _, predictions = torch.max(outputs, 1)
    acc = torch.sum(predictions.data == var_targets.data) / float(len(predictions))
    results['loss'] = criterion(outputs, var_targets).data[0]
    results['acc'] = acc
    return results

def eval_regression_model(net, graphs, targets, criterion):
    '''
    Args:
        net: a nn.Module
        graphs: list of Graph objects
        targets: a torch.Tensor or numpy array
    Returns:
        dictionary with rmse and mae as keys.
    '''
    if not isinstance(targets, (torch.DoubleTensor, torch.FloatTensor, torch.Tensor)):
        targets = torch.Tensor(targets)
    outputs = net(graphs)
    diff = outputs.data.squeeze() - targets

    results = {}
    results['rmse'] = np.sqrt(torch.mean(diff.mul(diff)))
    results['mae'] = torch.mean(torch.abs(diff))
    results['loss'] = criterion(outputs, targets).data[0]
    return results

def eval_model(net, graphs, targets, criterion):
    if net.mode == 'classification':
        return eval_classification_model(net, graphs, targets, criterion)
    elif net.mode == 'regression':
        return eval_regression_model(net, graphs, targets, criterion)
    else:
        raise ValueError("Need to pass in a network with mode = classification or regression")

def logline(line, logfile):
    '''
    Args:
        line: string message to print out to console
        logfile: string of filename to write the input line into
    '''
    line = time.strftime('[%H:%M:%S] ') + str(line)
    print(line)

    # dont write to file if log file is given
    if logfile is None:
        return

    with open(logfile, 'a') as f:
        if line[-1] != '\n':
            line = line + '\n'
        f.write(line)

def get_logger(logfile=None):
    '''
    Creates a logger function that can be called to print things to console and
    save the printed lines to the given logfile.
    Args:
        logfile: string of file name that the logger should write to
    Returns:
        a function
    '''
    log = lambda x: logline(x, logfile)
    return log

def get_args():
    '''
    Parse command line arguments with argparse
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="dataset", type=str, help="file path of the pickled dataset",
                        default='data/qm9.pickle')
    parser.add_argument("-lf", dest="logfile", type=str, help="name of the file to log to",
                        default=None)
    parser.add_argument("-l", dest="levels", type=int, help="number of layers", default=2)
    parser.add_argument("-hi",dest="hidden", type=int, help="size of hidden layers", default=5)
    parser.add_argument("-bs", dest="batchsize", type=int, help="batch size", default=1)
    parser.add_argument("-e", dest="max_epochs", type=int, help="max epochs", default=10)
    parser.add_argument("-me", dest="min_epochs", type=int, help="min epochs", default=2)
    parser.add_argument("-lr",dest="learning_rate", type=float, help="initial learning rate",
                        default=0.001)
    parser.add_argument("-tf", dest="train_frac", type=float, help="train fraction",default=0.1)
    parser.add_argument("-vf", dest="val_frac", type=float, help="val fraction",default=0.1)
    parser.add_argument("-s",dest="seed", type=int, help="random seed", default=42)

    return parser.parse_args()

def get_num_features(dataset_file):
    if 'MUTAG' in dataset_file:
        return 7
    if 'PTC' in dataset_file:
        return 22
    if 'PROTEINS' in dataset_file:
        return 3
    if 'NCI109' in dataset_file:
        return 38
    if 'NCI1' in dataset_file:
        return 37
    if 'qm9' in dataset_file or 'QM9' in dataset_file:
        return 5

def classify_or_regress(dataset_file):
    regression_datasets = ['qm9', 'QM9']
    for d in regression_datasets:
        if d in dataset_file:
            return 'regression'

    return 'classification'
def get_criterion(task):
    if task == 'regression':
        return nn.MSELoss()
    else:
        return nn.NLLLoss()

def make_model(hidden_size, levels, dataset_file):
    # need to know the number of input features for initializing the network
    nfeatures = get_num_features(dataset_file)
    task = classify_or_regress(dataset_file)
    #model = MolecFingerprintNet(levels, nfeatures, hidden_size, F.relu)
    model = MolecFingerprintNet_Adj(levels, nfeatures, hidden_size, F.relu, mode=task)

    return model

def epoch_results_str(epoch, train_results, val_results):
    '''
    Constructs a formatted string with epoch training/validation results
    Args:
        epoch: int of the epoch number
        train_results: dict with the keys mae and rmse
        val_results: dict with the keys mae and rmse
    Returns:
        formatted string
    '''
    #result =  "Epoch {} | train rmse: {:.5f} train mae: {:.5f} "
    train_loss = train_results.pop('loss')
    val_loss = val_results.pop('loss')
    result =  "Epoch {} | train: {} | val: {}".format(epoch, train_results, val_results)
    #result += "| val rmse: {:.5f} val mae: {:.5f}"
    #result = result.format(epoch, train_results['rmse'], train_results['mae'],
    #                       val_results['rmse'], val_results['mae'])
    train_results['loss'] = train_loss
    val_results['loss'] = val_loss
    return result

def main():
    args = get_args()
    log = get_logger(args.logfile)

    # data is a dict of train/validation/test graphs, and train/val/test target values
    # keys: graphs_train, graphs_val, graphs_test, y_train, y_val, y_test
    # graphs_train/graphs_val/graphs_test: list of Graph objects
    # y_train/y_val/y_test: numpy arrays
    log("Args are: {}".format(args))
    log("Starting to load data from: {}".format(args.dataset))
    data = train_val_test_dataset(args.dataset, train_frac=args.train_frac,
                                  val_frac=args.val_frac, seed=42)
    log("Done loading data")
    model = make_model(args.hidden, args.levels, args.dataset)
    log(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = get_criterion(model.mode)

    prev_val_loss = float('inf')
    log("Starting training...")
    for epoch in range(args.max_epochs):
        for i in range(0, len(data['graphs_train']), args.batchsize):
            g_batch = data['graphs_train'][i:i+args.batchsize]
            if model.mode == 'classification':
                y_batch = torch.LongTensor(data['y_train'][i:i+args.batchsize].astype(int))
            else:
                y_batch = torch.Tensor(data['y_train'][i:i+args.batchsize])

            optimizer.zero_grad()
            outputs = model.forward(g_batch)
            loss = criterion(outputs, Variable(y_batch, requires_grad=False))
            loss.backward()
            optimizer.step()

            # Print some things during batches of an epoch for debug purposes
            #if i % 1000 == 0 and i > 0:
            #    t_res = eval_model(model, g_batch, y_batch)
            #    log('   Epoch {} | Batch {} | rmse: {:.5f} | mae: {:.5f}'.format(epoch, i,
            #        t_res['rmse'], t_res['mae']))

        train_res = eval_model(model, data['graphs_train'], data['y_train'], criterion)
        val_res = eval_model(model, data['graphs_val'], data['y_val'], criterion)
        log(epoch_results_str(epoch, train_res, val_res))

        # Stop training if the validation error stops decreasing
        if val_res['loss'] > prev_val_loss and epoch > args.min_epochs:
            break

        data['graphs_train'], data['y_train'] = shuffle(data['graphs_train'],
                                                        data['y_train'],
                                                        random_state=args.seed)
        prev_val_loss = val_res['loss']

    log("Done training. Evaluating model on test data...")
    test_result = eval_model(model, data['graphs_test'], data['y_test'], criterion)
    log('Test result: {}'.format(test_result))

if __name__ == '__main__':
    main()
