#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


t0 = 0.7
t1 = 0.3
noise = 0.2


if __name__ == '__main__':
    x = np.random.rand(100)
    error = np.random.rand(100) * noise
    y = t0 + t1 * x + error
    t0e, t1e = np.dot(np.linalg.inv([[len(x), np.sum(x)], [np.sum(x), np.sum(x*x)]]), [np.sum(y), np.sum(x * y)])
    plt.plot(x, y, 'o')
    plt.plot(np.arange(2), t0e+np.arange(2)*t1e)
    plt.savefig('least_squares.pdf')
