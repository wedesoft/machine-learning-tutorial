#!/usr/bin/env python3
# https://gist.github.com/karpathy/d4dee566867f8291f086
import pytest
from numpy.testing import assert_array_equal
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class CharVec:
    def __init__(self, text):
        self.chars = np.array([ord(c) for c in  sorted(set(text))])

    def vector(self, c):
        return np.where(self.chars == ord(c), 1, 0).astype(np.float32)

    def index(self, arr):
        return ''.join([chr(self.chars[i]) for i in arr])

    def __call__(self, x):
        return np.array([self.vector(c) for c in x])

    def __len__(self):
        return len(self.chars)


class TestCharVec:
    def test_single_character(self):
        assert_array_equal(CharVec('a')('a'), [[1]])

    def test_first_character(self):
        assert_array_equal(CharVec('ab')('a'), [[1, 0]])

    def test_second_character(self):
        assert_array_equal(CharVec('ab')('b'), [[0, 1]])

    def test_multiple(self):
        assert_array_equal(CharVec('abx')('ax'), [[1, 0, 0], [0, 0, 1]])

    def test_remove_duplicates_from_ground_set(self):
        assert_array_equal(CharVec('aab')('b'), [[0, 1]])

    def test_dtype(self):
        assert CharVec('a')('a').dtype == np.float32

    def test_length(self):
        assert len(CharVec('abc')) == 3

    def test_index(self):
        assert CharVec('ab').index([1]) == 'b'


class RNN:
    def __init__(self, n, n_hidden):
        self.x = tf.placeholder(tf.float32, [1, n], name='x')
        self.y = tf.placeholder(tf.float32, [1, n], name='y')
        self.h = tf.placeholder(tf.float32, [1, n_hidden], name='h')
        self.wh = tf.Variable(tf.random_normal([n, n_hidden], stddev=1.0/n))
        self.uh = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=1.0/n_hidden))
        self.bh = tf.Variable(tf.constant(0.0, shape=[n_hidden]))
        self.wy = tf.Variable(tf.random_normal([n_hidden, n], stddev=1.0/n_hidden))
        self.by = tf.Variable(tf.constant(0.0, shape=[n]))

    def theta(self):
        return [self.wh, self.uh, self.bh, self.wy, self.by]

    def __call__(self, x, h):
        z1 = tf.matmul(x, self.wh) + tf.matmul(h, self.uh) + self.bh
        h_ = tf.tanh(z1)
        z2 = tf.matmul(h_, self.wy) + self.by
        y_ = tf.nn.softmax(z2)
        return y_, h_


def safe_log(v):
    return tf.log(tf.clip_by_value(v, 1e-10, 1.0))


def shakespeare():
    # http://www.gutenberg.org/ebooks/100.txt.utf-8
    with open('shakespeare.txt', 'r') as f:
        return f.read()[7429:]


if __name__ == '__main__':
    txt = shakespeare()
    char_vec = CharVec(txt)
    n = len(char_vec)
    v = char_vec(txt[0:100])

    n_iterations = 100000
    n_hidden = 100

    m = 25
    alpha = 0.1
    rnn = RNN(n, n_hidden)
    x = tf.placeholder(tf.float32, [m, n], name='x')
    y = tf.placeholder(tf.float32, [m, n], name='y')
    h = rnn.h
    theta = rnn.theta()

    cost = 0
    error = 0
    h_ = h
    for i in range(m):
        y_, h_ = rnn(x[i:i+1], h)
        cost = cost - tf.reduce_sum(y[i:i+1] * safe_log(y_) + (1 - y[i:i+1]) * safe_log(1 - y_)) / m
        error += tf.reduce_sum(tf.square(y[i:i+1] - y_)) / m
    error = tf.sqrt(error)

    hnext = rnn(rnn.x, rnn.h)[1]
    dtheta = tf.gradients(cost, theta)
    step = [tf.assign(value, tf.subtract(value, tf.multiply(alpha, dvalue)))
            for value, dvalue in zip(theta, dtheta)]

    i = 0

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        j_train = n * 0.5
        progress = tqdm(range(n_iterations))
        p = 0
        state = np.zeros((1, n_hidden))
        for i in progress:
            train = {x: char_vec(txt[p:p+m]), h:state, y:char_vec(txt[p+1:p+1+m])}
            j_train = 0.999 * j_train + 0.001 * session.run(error, feed_dict=train)
            if i % 10 == 0:
                progress.set_description('%8.6f' % j_train)
            session.run(step, feed_dict=train)
            state = session.run(h_, feed_dict=train)
            p += m
        tf.add_to_collection('prob', rnn(rnn.x, rnn.h)[0])
        tf.add_to_collection('hnext', hnext)
        saver.save(session, 'rnn')
