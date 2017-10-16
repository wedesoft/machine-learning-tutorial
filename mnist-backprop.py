#!/usr/bin/env python3
import sys
import pytest
import pickle
import gzip
import numpy as np
from numpy.testing import assert_array_equal
import tensorflow as tf


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


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    print(len(training[1]), 'training samples')
    print(len(validation[1]), 'validation samples')
    print(len(testing[1]), 'testing samples')

    n = 100
    x, label = random_selection(100, *training)
    y = multi_class_label(label, 10)

    x_ = tf.constant(x, dtype=tf.float32)
    y_ = tf.constant(y, dtype=tf.float32)

    m_ = tf.Variable(tf.random_uniform([28 * 28, 10], minval=-0.5, maxval=0.5, dtype=tf.float32))
    b_ = tf.Variable(tf.random_uniform([         10], minval=-0.5, maxval=0.5, dtype=tf.float32))
    h_ = tf.sigmoid(tf.matmul(x_, m_) + b_)

    loss = tf.reduce_sum(tf.square(h_ - y_)) / n

    train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(500):
            print(sess.run(loss))
            sess.run(train)
