A collection of graph neural networks implemented in pytorch.
Currently a work in progress.


Usage:
    python run.py -l {LEVELS} -hi {HIDDEN} -bs {BATCHSIZE} -e {EPOCHS} -lr {LEARNING_RATE}

    This will run the molecular fingerprints regressor on the QM9 dataset.
    Modify the PICKLEFILE variable in run.py to point to the location of the QM9 pickle
    file.

To see all possible commandline arguments run:
    python run.py --help


References:
"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf
