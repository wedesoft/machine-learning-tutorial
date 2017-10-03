import numpy as np
import gzip
import pickle

# https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
train, valid, test = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
