import sys
import argparse
from utils.load_data import make_dataset_pickle
import time

gname = 'data/qm9.graph'
lname = 'data/qm9.atoms'
tname = 'data/qm9.target'
pickle_name = 'data/qm9.pickle'
start_time = time.time()
print("Starting to pickle...")
make_dataset_pickle(gname, lname, tname, pickle_name)
print("Done pickling!")
print("Time elapsed: {:.2f}".format(time.time() - start_time))

