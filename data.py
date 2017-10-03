#!/usr/bin/env python3
import pytest
import numpy as np
import gzip
import pickle


def average(data):
    return np.average(data)


class TestAverage:
    def test_single_element(self):
        assert average([42]) == 42

    def tesT_two_elements(self):
        assert average([10, 20]) == 15


def std(data):
    deviation = data - average(data)
    return average(deviation ** 2) ** 0.5


class TestStandardDeviation:
    def test_single_element(self):
        assert std([42]) == 0

    def test_two_elements(self):
        assert std([10, 20]) == 5


def covariance(x, y):
    return average((x - average(x)) * (y - average(y)))


class TestCovariance:
    def test_single_elements(self):
        assert covariance([42], [22]) == 0

    def test_linear(self):
        assert covariance([-1, 1], [-1, 1]) == 1

    def test_negative(self):
        assert covariance([-1, 1], [1, -1]) == -1


def correlation_coefficient(x, y):
    return covariance(x, y) / (std(x) * std(y))


class TestCorrelationCoefficient:
    def test_linear(self):
        assert correlation_coefficient([-1, 1], [-1, 1]) == 1

    def test_negative(self):
        assert correlation_coefficient([-1, 1], [1, -1]) == -1

    def test_normalisation(self):
        assert correlation_coefficient([-2, 2], [-2, 2]) == 1

    def test_normalise_both(self):
        assert correlation_coefficient([-2, 2], [-4, 4]) == 1


# camera -> connected components -> label

if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    print(len(training[1]), 'training samples')
    print(len(validation[1]), 'validation samples')
    print(len(testing[1]), 'testing samples')
