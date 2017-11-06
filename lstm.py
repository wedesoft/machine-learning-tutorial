#!/usr/bin/env python3
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


class LSTM:
    def __init__(self, n, n_hidden):
        # https://en.wikipedia.org/wiki/Long_short-term_memory
        self.x = tf.placeholder(tf.float32, [1, n], name='x')
        self.h = tf.placeholder(tf.float32, [1, n_hidden], name='h')
        self.c = tf.placeholder(tf.float32, [1, n_hidden], name='c')
        self.wf = tf.Variable(tf.random_normal([n, n_hidden], stddev=1.0/n))
        self.uf = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=1.0/n))
        self.bf = tf.Variable(tf.constant(0.0, shape=[n_hidden]))
        self.wi = tf.Variable(tf.random_normal([n, n_hidden], stddev=1.0/n))
        self.ui = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=1.0/n))
        self.bi = tf.Variable(tf.constant(0.0, shape=[n_hidden]))
        self.wo = tf.Variable(tf.random_normal([n, n_hidden], stddev=1.0/n))
        self.uo = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=1.0/n))
        self.bo = tf.Variable(tf.constant(0.0, shape=[n_hidden]))
        self.wc = tf.Variable(tf.random_normal([n, n_hidden], stddev=1.0/n))
        self.uc = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=1.0/n))
        self.bc = tf.Variable(tf.constant(0.0, shape=[n_hidden]))

    def theta(self):
        return [self.wf, self.uf, self.bf,
                self.wi, self.ui, self.bi,
                self.wo, self.uo, self.bo,
                self.wc, self.uc, self.bc]

    def __call__(self, x, h, c):
        f = tf.sigmoid(tf.matmul(x, self.wf) + tf.matmul(h, self.uf) + self.bf)
        i = tf.sigmoid(tf.matmul(x, self.wi) + tf.matmul(h, self.ui) + self.bi)
        o = tf.sigmoid(tf.matmul(x, self.wo) + tf.matmul(h, self.uo) + self.bo)
        g = tf.tanh(tf.matmul(x, self.wc) + tf.matmul(h, self.uc) + self.bc)
        c_ = f * c + i * g
        h_ = o * tf.tanh(c_)
        return h_, c_


def safe_log(v):
    return tf.log(tf.clip_by_value(v, 1e-10, 1.0))


def source_code():
	# http://www.gutenberg.org/cache/epub/100/pg100.txt
    with open('shakespeare.txt', 'r') as f:
        return f.read()[10462:113402]


if __name__ == '__main__':
    txt = source_code()
    print(txt[:200])
    print('...')
    print(txt[-200:])
    char_vec = CharVec(txt)
    n = len(char_vec)
    v = char_vec(txt[0:100])

    n_iterations = 500000

    m = 50
    n_hidden = 100
    alpha = 1.0
    lstm = LSTM(n, n_hidden)
    wy = tf.Variable(tf.random_normal([n_hidden, n], stddev=1.0/n))
    by = tf.Variable(tf.constant(0.0, shape=[n]))
    x = tf.placeholder(tf.float32, [m, n], name='x')
    y = tf.placeholder(tf.float32, [m, n], name='y')
    out = lambda h: tf.sigmoid(tf.matmul(h, wy) + by)
    theta = lstm.theta() + [wy, by]

    cost = 0
    error = 0
    h_, c_ = lstm.h, lstm.c
    for i in range(m):
        h_, c_ = lstm(x[i:i+1], h_, c_)
        h = out(h_)
        cost += -tf.reduce_sum(y[i:i+1] * safe_log(h) + (1 - y[i:i+1]) * safe_log(1 - h)) / m

    dtheta = tf.gradients(cost, theta)
    step = [tf.assign(value, tf.subtract(value, tf.multiply(alpha, dvalue)))
            for value, dvalue in zip(theta, dtheta)]

    i = 0

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        j_train = 0.0
        progress = tqdm(range(n_iterations))
        p = 0
        for i in progress:
            if p + m >= len(txt):
                p = 0
            if p == 0:
                h = np.zeros((1, n_hidden))
                c = np.zeros((1, n_hidden))
            train = {x: char_vec(txt[p:p+m]), lstm.c: c, lstm.h: h, y:char_vec(txt[p+1:p+1+m])}
            j_train = 0.999 * j_train + 0.001 * session.run(cost, feed_dict=train)
            if i % 10 == 0:
                progress.set_description('%8.6f' % j_train)
            session.run(step, feed_dict=train)
            h = session.run(h_, feed_dict=train)
            c = session.run(c_, feed_dict=train)
            p += m
        tf.add_to_collection('hnext', lstm(lstm.x, lstm.h, lstm.c)[0])
        tf.add_to_collection('cnext', lstm(lstm.x, lstm.h, lstm.c)[1])
        tf.add_to_collection('prob', out(lstm(lstm.x, lstm.h, lstm.c)[0]))
        saver.save(session, './lstm')
