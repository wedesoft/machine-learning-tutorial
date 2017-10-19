#!/usr/bin/env python3
# http://blog.aloni.org/posts/backprop-with-tensorflow/
import pytest
from numpy.testing import assert_array_equal
import sys
import numpy as np
import tensorflow as tf
import pickle
import gzip
from tqdm import tqdm


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


def multi_class_label(labels, num_classes):
    index = np.arange(len(labels) * num_classes).reshape(len(labels), num_classes)
    return np.where(np.equal(index % num_classes, np.expand_dims(labels, -1)), 1, 0)


class TestMultiClassLabel:
    def test_one_class_label(self):
        assert_array_equal(multi_class_label([0], 1), [[1]])

    def test_two_class_label(self):
        assert_array_equal(multi_class_label([1], 2), [[0, 1]])

    def test_first_class_of_two(self):
        assert_array_equal(multi_class_label([0], 2), [[1, 0]])

    def test_two_samples(self):
        assert_array_equal(multi_class_label([1, 0], 3), [[0, 1, 0], [1, 0, 0]])

    def test_return_type(self):
        assert multi_class_label([0], 1).dtype != np.bool


class Scale:
    def __init__(self, features, max_scale=10.0):
        self.average = np.average(features, axis=0)
        self.deviation = np.maximum(np.std(features, axis=0), 1 / max_scale)

    def __call__(self, values):
        return np.subtract(values, self.average) / self.deviation


class TestFeatureScaling:
    def test_basic_average(self):
        assert_array_equal(Scale([[5]], 100).average, [5])

    def test_average_of_two_samples(self):
        assert_array_equal(Scale([[5], [7]], 100).average, [6])

    def test_vector_of_averages(self):
        assert_array_equal(Scale([[2, 3]], 100).average, [2, 3])

    def test_lower_bound_deviation(self):
        assert_array_equal(Scale([[5]], 100).deviation, [0.01])

    def test_standard_deviation_of_two_samples(self):
        assert_array_equal(Scale([[5], [7]], 100).deviation, [1])

    def test_vector_of_deviations(self):
        assert_array_equal(Scale([[2, 3], [2, 5]], 100).deviation, [0.01, 1])

    def test_subtract_average(self):
        assert_array_equal(Scale([[5], [7]], 100)([[9], [10]]), [[3], [4]])

    def test_normalise_standard_deviation(self):
        assert_array_equal(Scale([[4], [8]], 100)([[6], [8]]), [[0], [1]])

    def test_normalise_feature_vector(self):
        assert_array_equal(Scale([[0, 0], [2, 4]], 100)([[0, 0], [1, 2], [2, 4]]), [[-1, -1], [0, 0], [1, 1]])

    def test_limit_scaling(self):
        assert_array_equal(Scale([[0]], 100)([[1]]), [[100]])


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')

    n_samples = 10000
    alpha = tf.constant(1.0)
    training = random_selection(n_samples, *training)
    scale = Scale(training[0], 1000)
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    m1 = tf.Variable(tf.truncated_normal([784, 10], stddev=1/784))
    b1 = tf.Variable(tf.truncated_normal([10]))

    a0 = tf.sigmoid(x)
    z1 = tf.add(tf.matmul(a0, m1), b1)
    a1 = tf.sigmoid(z1)
    h = a1

    cost = -tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h))
    dm1 = tf.gradients(cost, m1)
    db1 = tf.gradients(cost, b1)

    step = [
            tf.assign(m1, tf.subtract(m1, tf.multiply(alpha, dm1[0]))),
            tf.assign(b1, tf.subtract(b1, tf.multiply(alpha, db1[0])))
            ]

    with tf.Session() as sess:
        train = {x: scale(training[0] ), y: multi_class_label(training[1], 10)}
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(1000)):
            sess.run(step, feed_dict=train)
        validate = {x: scale(validation[0] ), y: multi_class_label(validation[1], 10)}
        prediction = np.argmax(sess.run(h, feed_dict=validate), axis=-1)
        print(validation[1])
        print(prediction)
        print(np.average(prediction != validation[1]))
