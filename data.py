#!/usr/bin/env python3
import pytest
import gzip
import pickle
import numpy as np
from numpy.testing import assert_array_equal
import cv2


def average(data):
    return np.average(data, axis=-1)


class TestAverage:
    def test_single_element(self):
        assert average([42]) == 42

    def test_two_elements(self):
        assert average([10, 20]) == 15

    def test_batch(self):
        assert_array_equal(average([[5], [7]]), [5, 7])


def deviation(data):
    return np.transpose(np.transpose(data) - average(data))


def std(data):
    return average(deviation(data) ** 2) ** 0.5


class TestStandardDeviation:
    def test_single_element(self):
        assert std([42]) == 0

    def test_two_elements(self):
        assert std([10, 20]) == 5

    def test_batch(self):
        assert_array_equal(std([[3], [5]]), [0, 0])


def covariance(x, y):
    return average(deviation(x) * deviation(y))


class TestCovariance:
    def test_single_elements(self):
        assert covariance([42], [22]) == 0

    def test_linear(self):
        assert covariance([-1, 1], [-1, 1]) == 1

    def test_negative(self):
        assert covariance([-1, 1], [1, -1]) == -1

    def test_batch(self):
        assert_array_equal(covariance([[-1, 1], [-1, 1], [1, -1]], [-1, 1]), [1, 1, -1])


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

    def test_batch(self):
        assert_array_equal(correlation_coefficient([[-1, 1], [1, -1], [-1, 1]], [-1, 1]), [1, -1, 1])


def show(img, wait=True):
    cv2.imshow('show', cv2.resize(img.reshape(28, 28), (280, 280)))
    return cv2.waitKey(-1 if wait else 1) != 27


def save(file_name, img):
    cv2.imwrite(file_name, img.reshape(28, 28))


def index(data, value):
    return np.where(data == value)[0][0]


# camera -> connected components -> label

if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    print(len(training[1]), 'training samples')
    print(len(validation[1]), 'validation samples')
    print(len(testing[1]), 'testing samples')
    n = 0
    error = 0
    for test, label in zip(testing[0], testing[1]):
        result = training[1][np.argmax(correlation_coefficient(training[0], test))]
        print("%d (%d)" % (result, label))
        if result != label:
            save('%03d_%d.png' % (n, result), test)
            error += 1
        n += 1
        print('error rate %5.2f%%' % (100 * error / n))
        if not show(test, False):
            break
