#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


def gradient_function():
    x, y, t = T.vectors('x', 'y', 't')
    h = t[0] + t[1] * x
    cost = T.sqr((h - y)).sum() / (2 * x.size)
    dt = T.grad(cost, t)
    return theano.function([x, y, t], outputs=dt)


if __name__ == '__main__':
    t = [0.2, 0.5]
    x = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + t[1] * x + error
    plt.plot(x, y, 'bo', label='data')
    plt.plot(np.arange(2), t[0] + np.arange(2) * t[1], 'r', label='ground truth')

    alpha = 0.5
    gradient = gradient_function()
    t = [0, 0]

    n = 100
    for i in range(1, n + 1):
        dt = gradient(x, y, t)
        t -= alpha * dt
        if i % 10 == 0:
            plt.plot(np.arange(2), t[0] + np.arange(2) * t[1], color=(0, 0.5, 0, float(i) / n), label="iteration %3d" % i)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent (alpha=%4.2f)' % alpha)
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('gradient_descent.pdf')
