#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


def cost(x, y, t):
    return T.sqr((t[0] + t[1] * x - y)).sum() / (2 * x.size)

def gradient_function():
    x, y, t = T.vectors('x', 'y', 't')
    return theano.function([x, y, t], outputs=T.grad(cost(x, y, t), t))

def cost_function():
    x, y, t = T.vectors('x', 'y', 't')
    return theano.function([x, y, t], outputs=cost(x, y, t))


if __name__ == '__main__':
    t = [0.2, 0.5]
    x = np.random.rand(100)
    error = np.random.rand(100) * 0.2 - 0.1
    y = t[0] + t[1] * x + error

    gradient = gradient_function()
    cost = cost_function()

    n = 100
    alphas = [0.5 * 1.5 ** step for step in range(-2, 5)]
    for alpha in alphas:
        t = [0, 0]
        c = []
        for i in range(1, n + 1):
            c.append(cost(x, y, t))
            dt = gradient(x, y, t)
            t -= alpha * gradient(x, y, t)
        plt.semilogy(range(n), c, label='alpha=%4.2f' % alpha)

    plt.title('Convergence depending on learning rate')
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.legend()
    plt.axis([0, 100, 0, 0.2])
    plt.savefig('learning_rate.pdf')
