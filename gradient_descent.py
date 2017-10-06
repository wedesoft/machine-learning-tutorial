#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


def gradient_function():
    x = T.vector('x')
    y = T.vector('y')
    t0 = T.scalar('t0')
    t1 = T.scalar('t1')
    cost = T.sqr((t0 + t1 * x - y)).sum() / (2 * x.size)
    dt0 = T.grad(cost, t0)
    dt1 = T.grad(cost, t1)
    return theano.function([x, y, t0, t1], outputs=(dt0, dt1))


if __name__ == '__main__':
    x = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = 0.2 + 0.5 * x + error
    plt.plot(x, y, 'o', label='data')

    alpha = 0.5
    gradient = gradient_function()
    t0, t1 = 0.0, 0.0

    for i in range(10):
        dt0, dt1 = gradient(x, y, t0, t1)
        t0 -= alpha * dt0
        t1 -= alpha * dt1
        plt.plot(np.arange(2), t0 + np.arange(2) * t1, color=(0, 0.5, 0, i * 0.1 + 0.1), label="iteration %d" % i)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('gradient_descent.pdf')
