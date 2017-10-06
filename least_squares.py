#!/usr/bin/env python3
import numpy as np
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = 0.7 + 0.3 * x + error
    plt.plot(x, y, 'o')

    t0, t1 = np.dot(np.linalg.inv([[len(x), np.sum(x)], [np.sum(x), np.sum(x*x)]]), [np.sum(y), np.sum(x * y)])
    plt.plot(np.arange(2), t0 + np.arange(2) * t1)

    plt.savefig('least_squares.pdf')
