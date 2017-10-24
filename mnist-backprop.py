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
    try:
        training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    except TypeError:
        training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'))

    n_samples = 50000
    n_classes = 10
    n_iterations = 4000
    n_hidden = 600
    regularize = 0.2
    alpha = 0.5
    training = random_selection(n_samples, *training)
    scale = Scale(training[0], 1000)

    x = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
    y = tf.placeholder(tf.float32, [None, 10])
    m1 = tf.Variable(tf.truncated_normal([784, n_hidden], stddev=1/784))
    b1 = tf.Variable(tf.truncated_normal([n_hidden]))
    m2 = tf.Variable(tf.truncated_normal([n_hidden, 10], stddev=1/n_hidden))
    b2 = tf.Variable(tf.truncated_normal([10]))
    theta = [m1, b1, m2, b2]

    a0 = tf.sigmoid(x)
    z1 = tf.add(tf.matmul(a0, m1), b1)
    a1 = tf.sigmoid(z1)
    z2 = tf.add(tf.matmul(a1, m2), b2)
    a2 = tf.sigmoid(z2)
    h = a2

    prediction = tf.argmax(h, axis=-1)

    m = tf.cast(tf.size(y) / n_classes, tf.float32)
    reg_term = (tf.reduce_sum(tf.square(m1)) + tf.reduce_sum(tf.square(m2))) * regularize / (m * 2)
    error_term = -tf.reduce_sum(y * tf.log(h) + (1 - y) * tf.log(1 - h)) / m
    cost = error_term + regularize * reg_term
    rmsd = tf.reduce_sum(tf.square(h - y)) / (2 * m)
    dtheta = tf.gradients(cost, theta)

    step = [tf.assign(value, tf.subtract(value, tf.multiply(alpha, dvalue)))
            for value, dvalue in zip(theta, dtheta)]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train = {x: scale(training[0] ), y: multi_class_label(training[1], n_classes)}
        validate = {x: scale(validation[0] ), y: multi_class_label(validation[1], n_classes)}
        test = {x: scale(testing[0] ), y: multi_class_label(testing[1], n_classes)}
        sess.run(tf.global_variables_initializer())
        progress = tqdm(range(n_iterations))
        info = lambda: 'train: %8.6f, validate: %8.6f, test: %8.6f' % \
                (sess.run(rmsd, feed_dict=train),
                 sess.run(rmsd, feed_dict=validate),
                 sess.run(rmsd, feed_dict=test))
        for i in progress:
            sess.run(step, feed_dict=train)
            if i % 10 == 0:
                progress.set_description(info())
        print(info())
        output = sess.run(prediction, feed_dict=validate)
        print('validation labels:', validation[1])
        print('predictions      :', output)
        print('error rate:', np.average(output != validation[1]))
        tf.add_to_collection('prediction', prediction)
        saver.save(sess, 'model')
