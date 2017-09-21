from six.moves import cPickle as pickle
import pdb

picklefile = 'gabor_total_energy.pickle'
picklefile = 'gabor_avg_energy.pickle'

with open(picklefile, 'r') as p1:
    total_energy = pickle.load(p1)
    pdb.set_trace()
