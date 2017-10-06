#!/usr/bin/env python3
import numpy as np
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


if __name__ == '__main__':
    t = [0.2, 0.5]
    x = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + t[1] * x + error
    plt.plot(x, y, 'bo', label='data')
    plt.plot(np.arange(2), t[0] + np.arange(2) * t[1], 'r', label='ground truth')

    t0, t1 = np.dot(np.linalg.inv([[len(x), np.sum(x)], [np.sum(x), np.sum(x*x)]]), [np.sum(y), np.sum(x * y)])
    plt.plot(np.arange(2), t0 + np.arange(2) * t1, label='result')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Estimate')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('least_squares.pdf')
