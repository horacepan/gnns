import sys
import argparse
from utils.load_data import make_dataset_pickle
import time

for dataset in ['NCI1', 'NCI109', 'PTC', 'PROTEINS', 'MUTAG']:
    gname = 'data/{}.graph'.format(dataset)
    lname = 'data/{}.node'.format(dataset)
    tname = 'data/{}.label'.format(dataset)
    pickle_name = 'data/{}.pickle'.format(dataset)

    start_time = time.time()
    print("Starting to pickle...")
    make_dataset_pickle(gname, lname, tname, pickle_name)
    print("Done pickling!")
    print("Time elapsed: {:.2f}".format(time.time() - start_time))

