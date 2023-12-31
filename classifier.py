#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


def gradient_function():
    x = T.matrix('x')
    y, t = T.vectors('y', 't')
    h = 1 / (1 + T.exp(-theano.dot(t, x)))
    cost = -(y * T.log(h) + (1 - y) * T.log(1 - h)).sum() / y.size
    dt = T.grad(cost, t)
    return theano.function([x, y, t], outputs=dt)


if __name__ == '__main__':
    t = [-0.3, 0.2, 0.5]
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + x1 * t[1] + x2 * t[2] >= error

    x0 = np.full(x1.shape, 1)
    alpha = 4.0
    gradient = gradient_function()
    t = [0, 0, 0]
    n = 100
    for i in range(1, n + 1):
        dt = gradient([x0, x1, x2], y, t)
        t -= alpha * dt
        if i % 10 == 0:
            C = np.arange(0, 1, 0.01)
            X1, X2 = np.meshgrid(C, C)
            plt.contour(X1, X2, t[0] + X1 * t[1] + X2 * t[2], levels = [0], colors=((0, 0.5, 0, float(i) / n),), label="iteration %3d" % i)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gradient Descent (alpha=%4.2f, n=%d)' % (alpha, n))
    plt.scatter(np.compress(y, x1), np.compress(y, x2), color='r', marker='x', label='y=1')
    plt.scatter(np.compress(np.logical_not(y), x1), np.compress(np.logical_not(y), x2), color='b', marker='o', label='y=0')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.savefig('classifier.pdf')
