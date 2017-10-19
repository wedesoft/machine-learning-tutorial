#!/usr/bin/env python3
import operator
from functools import reduce
import sys
import pytest
import pickle
import gzip
import numpy as np
from numpy.testing import assert_array_equal
from tqdm import tqdm
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


def tensor(value):
    return tf.constant(value, dtype=tf.float32)


def random_tensor(*shape):
    scale = 1
    return tf.Variable(tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32))


def data(scale, features, labels):
    return scale(features), multi_class_label(labels, 10)


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')

    n_samples = 50000
    training = random_selection(n_samples, *training)
    validation = random_selection(n_samples // 5, *validation)
    testing = random_selection(n_samples // 5, *testing)
    print(len(training[1]), 'training samples')
    print(len(validation[1]), 'validation samples')
    print(len(testing[1]), 'testing samples')

    scale = Scale(training[0], 100)

    n_iterations = 1500
    n_hidden1 = 200
    n_hidden2 = 100
    alpha = 1.0
    regularization = 0.004 # validation: 0.057019
    regularization = 0.008 # validation: 0.054460
    regularization = 0.016 # validation: 0.055690
    regularization = 0.032 # validation: 0.054299
    regularization = 0.064 # validation: 0.057817
    regularisation = 0.032 # test: 0.060644
    learning_curve_samples = 1

    for n in [n_samples // 2 ** e for e in reversed(range(learning_curve_samples))]:
        X = tf.placeholder("float", name='X')
        Y = tf.placeholder("float", name='Y')
        x_train, y_train = data(scale, training[0][0:n], training[1][0:n])
        x_validation, y_validation = data(scale, *validation)
        x_testing, y_testing = data(scale, *testing)

        m1 = random_tensor(28 * 28, n_hidden1)
        b1 = random_tensor(n_hidden1)
        h1 = tf.sigmoid(tf.matmul(X, m1) + b1)
        m2 = random_tensor(n_hidden1, n_hidden2)
        b2 = random_tensor(n_hidden2)
        h2 = tf.sigmoid(tf.matmul(h1, m2) + b2)
        m3 = random_tensor(n_hidden2, 10)
        b3 = random_tensor(10)
        h = tf.sigmoid(tf.matmul(h2, m3) + b3, name='h')

        predict_op = tf.argmax(h, 1)
        tf.add_to_collection('predict_op', predict_op)

        cost_train = -tf.reduce_sum(Y * tf.log(h) + (1 - Y) * tf.log(1 - h)) / n + \
                      regularization * (tf.reduce_sum(tf.square(m1)) + tf.reduce_sum(tf.square(m2))) / (2 * n)
        loss = tf.reduce_sum(tf.square(h - Y)) / (tf.cast(tf.size(Y), tf.float32) / 10)

        train = tf.train.GradientDescentOptimizer(alpha).minimize(cost_train)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver([m1, b1, m2, b2, m3, b3])

        with tf.Session() as sess:
            sess.run(init_op)
            progress = tqdm(range(n_iterations))
            for step in progress:
                sess.run(train, feed_dict={X: x_train, Y: y_train})
            print("samples: %d, train: %f, validation: %f, testing: %f" % \
                    (n,
                     sess.run(loss, feed_dict={X: x_train     , Y: y_train     }),
                     sess.run(loss, feed_dict={X: x_validation, Y: y_validation}),
                     sess.run(loss, feed_dict={X: x_testing   , Y: y_testing   })))
            print('saved model as', saver.save(sess, "mnist.ckpt"))
