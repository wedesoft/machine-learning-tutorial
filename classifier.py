#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


if __name__ == '__main__':
    t = [-0.3, 0.2, 0.5]
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + x1 * t[1] + x2 *t[2] >= error
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(np.compress(y, x1), np.compress(y, x2), color='r', marker='x', label='y=1')
    plt.scatter(np.compress(np.logical_not(y), x1), np.compress(np.logical_not(y), x2), color='b', marker='o', label='y=0')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('classifier.pdf')
