#!/usr/bin/env python3
# https://github.com/craffel/theano-tutorial/blob/master/Backpropagation.ipynb
import pytest
import math
from numpy.testing import assert_array_equal
import numpy as np
import theano
import theano.tensor as T


class Layer:
    def __init__(self, weight, bias, activation=None):
        self.weight_ = theano.shared(value=np.array(weight), name='weight')
        self.bias_ = theano.shared(value=np.array(bias), name='bias')
        self.g_ = self.sigmoid_function if activation is None else activation
        x_ = T.vector('x')
        self.fun = theano.function([x_], outputs=self.output(x_))

    def output(self, x_):
        z_ = T.dot(self.weight_, x_) + self.bias_
        a_ = self.g_(z_)
        return a_

    @staticmethod
    def sigmoid_function(z_):
        return 1 / (1 + T.exp(-z_))

    def __call__(self, x):
        return self.fun(x)


class TestLayer:
    @pytest.fixture
    def z(self):
        return T.scalar('z')

    @pytest.fixture
    def sigmoid(self, z):
        return theano.function([z], outputs=Layer.sigmoid_function(z))

    def test_sigmoid(self, sigmoid):
        assert sigmoid(0) == 0.5
        assert abs(sigmoid(+10) - 1) < 1e-3
        assert abs(sigmoid(-10) - 0) < 1e-3

    def test_zero(self):
        assert_array_equal(Layer([[0, 0, 0]], [0], lambda z: z)([0, 0, 0]), [0])

    def test_bias(self):
        assert_array_equal(Layer([[0, 0, 0]], [3], lambda z: z)([0, 0, 0]), [3])

    def test_weights(self):
        assert_array_equal(Layer([[1, 2, 3]], [0], lambda z: z)([1, 1, 1]), [6])

    def test_use_sigmoid_by_default(self):
        assert_array_equal(Layer([[1, 2, 3]], [-23])([2, 3, 5]), [0.5])


class MLP:
    def __init__(self, weights, biases):
        self.layers = [Layer(weight, bias) for weight, bias in zip(weights, biases)]

    @staticmethod
    def logistic_cost(h, y):
        return -(math.log(h) * y + (1 - y) * math.log(1 - h))

    def __call__(self, x):
        retval = x
        for layer in self.layers:
            retval = layer(retval)
        return retval


class TestMLP:
    @pytest.fixture
    def layer1(self):
        return Layer([[2, 3, 5]], [5])

    @pytest.fixture
    def layer2(self):
        return Layer([[2]], [1])

    def test_single_layer(self, layer1):
        assert_array_equal(MLP([[[2, 3, 5]]], [[5]])([1, 2, -3]), layer1([1, 2, -3]))

    def test_two_layers(self, layer1, layer2):
        assert_array_equal(MLP([[[2, 3, 5]], [[2]]], [[5], [1]])([1, 2, -3]), layer2(layer1([1, 2, -3])))

    def test_logistic_cost(self):
        assert MLP.logistic_cost(0.1, 0) > 0
        assert MLP.logistic_cost(0.1, 0) < 0.2
        assert MLP.logistic_cost(0.9, 1) == MLP.logistic_cost(0.1, 0)
        assert MLP.logistic_cost(0.9, 0) > 2


if __name__ == '__main__':
    pass
