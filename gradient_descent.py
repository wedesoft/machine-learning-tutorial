#!/usr/bin/env python3
import theano
import theano.tensor as T


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
    y = 0.7 + 0.3 * x + error

    alpha = 1.0
    gradient = gradient_function()
    t0, t1 = 0, 0

    dt0, dt1 = gradient(x, y, t0, t1)
    t0 -= alpha * dt0
    t1 -= alpha * dt1
