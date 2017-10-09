#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


def gradient_function():
    x1, x2, y, t = T.vectors('x1', 'x2', 'y', 't')
    h = 1 / (1 + T.exp(-t[0] - t[1] * x1 - t[2] * x2))
    cost = -(y * T.log(h) + (1 - y) * T.log(1 - h)).sum() / x1.size
    dt = T.grad(cost, t)
    return theano.function([x1, x2, y, t], outputs=dt)

if __name__ == '__main__':
    t = [-0.3, 0.2, 0.5]
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + x1 * t[1] + x2 *t[2] >= error

    alpha = 4.0
    gradient = gradient_function()
    t = [0, 0, 0]
    n = 100
    for i in range(1, n + 1):
        dt = gradient(x1, x2, y, t)
        t -= alpha * dt
        if i % 10 == 0:
            plt.plot(np.arange(2), -(t[0] + np.arange(2) * t[1]) / t[2], color=(0, 0.5, 0, float(i) / n), label="iteration %3d" % i)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gradient Descent (alpha=%4.2f)' % alpha)
    plt.scatter(np.compress(y, x1), np.compress(y, x2), color='r', marker='x', label='y=1')
    plt.scatter(np.compress(np.logical_not(y), x1), np.compress(np.logical_not(y), x2), color='b', marker='o', label='y=0')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('classifier.pdf')