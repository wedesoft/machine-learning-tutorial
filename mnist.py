#!/usr/bin/env python3
import sys
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


def distance(a, b):
    return average(np.absolute(np.subtract(b, a)))


class TestDistance:
    def test_single_element(self):
        assert distance([2], [5]) == 3

    def test_two_elements(self):
        assert distance([0, 2], [2, 8]) == 4

    def test_absolute_difference(self):
        assert distance([2, 0], [0, 2]) == 2

    def test_batch(self):
        assert_array_equal(distance([[2], [3]], [[5], [7]]), [3, 4])


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


def random_choice(count, size):
    return np.random.choice(count, size, replace=False)


class TestRandomChoice:
    def test_select_one_of_one(self):
        assert_array_equal(random_choice(1, 1), [0])

    def test_select_one_of_ten(self):
        result = random_choice(10, 1)
        assert result[0] >= 0 and result[0] < 10

    def test_select_distinct_values(self):
        assert sorted(random_choice(10, 10)) == list(range(10))


def random_selection(size, *arrays):
    indices = random_choice(len(arrays[0]), size)
    result = tuple(np.take(array, indices, axis=0) for array in arrays)
    return result[0] if len(result) == 1 else result


class TestRandomSelection:
    @pytest.fixture
    def choose_first(self, monkeypatch):
        monkeypatch.setattr(sys.modules[__name__], 'random_choice', lambda count, size: range(size))

    @pytest.fixture
    def choose_last(self, monkeypatch):
        monkeypatch.setattr(sys.modules[__name__], 'random_choice', lambda count, size: range(count - size, count))

    def test_select_one(self):
        assert_array_equal(random_selection(1, [42]), [42])

    def test_select_first_of_two(self, choose_first):
        assert_array_equal(random_selection(1, [3, 5]), [3])

    def test_select_second_of_two(self, choose_last):
        assert_array_equal(random_selection(1, [3, 5]), [5])

    def test_select_from_multiple_arrays(self, choose_first):
        result = random_selection(1, [1, 2, 3], [4, 5, 6])
        assert_array_equal(result[0], [1])
        assert_array_equal(result[1], [4])

    def test_select_from_2d_array(self):
        assert_array_equal(random_selection(1, [[1, 2, 3]]), [[1, 2, 3]])


def show(img, wait=True):
    cv2.imshow('show', cv2.resize(img.reshape(28, 28), (280, 280)))
    return cv2.waitKey(-1 if wait else 1) != 27


def save(file_name, img):
    cv2.imwrite(file_name, (img.reshape(28, 28) * 255).astype(np.uint8))


def index(data, value):
    return np.where(data == value)[0][0]


def make_classifier(images, labels):
    return lambda test: labels[np.argmin(distance(images, test))]


# camera -> connected components -> label

if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    print(len(training[1]), 'training samples')
    print(len(validation[1]), 'validation samples')
    print(len(testing[1]), 'testing samples')
    n = 0
    error = 0
    classifier = make_classifier(*random_selection(5000, *training))
    for test, label in zip(testing[0], testing[1]):
        result = classifier(test)
        print("%d (%d)" % (result, label))
        if result != label:
            save('%05d_%d.png' % (n, result), test)
            error += 1
        n += 1
        print('error rate %5.2f%%' % (100 * error / n))
        if not show(test, False):
            break
