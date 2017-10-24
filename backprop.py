#!/usr/bin/env python3
# https://stackoverflow.com/questions/44210561/how-do-backpropagation-works-in-tensorflow
# https://github.com/craffel/theano-tutorial/blob/master/Backpropagation.ipynb
import math
from numpy.testing import assert_array_equal
import numpy as np
import tensorflow as tf
from tqdm import tqdm


if __name__ == '__main__':
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    x0 = tf.constant(x, dtype=tf.float32)
    y0 = tf.constant(y, dtype=tf.float32)

    m1 = tf.Variable(tf.random_uniform([2, 2], minval=0.1, maxval=0.9, dtype=tf.float32))
    b1 = tf.Variable(tf.random_uniform([2]   , minval=0.1, maxval=0.9, dtype=tf.float32))
    h1 = tf.sigmoid(tf.matmul(x0, m1) + b1)

    m2 = tf.Variable(tf.random_uniform([2, 1], minval=0.1, maxval=0.9, dtype=tf.float32))
    b2 = tf.Variable(tf.random_uniform([1]   , minval=0.1, maxval=0.9, dtype=tf.float32))
    y_out = tf.sigmoid(tf.matmul(h1, m2) + b2)

    loss = tf.reduce_sum(tf.square(y0 - y_out))
    cost = -tf.reduce_mean(y0 * tf.log(y_out) + (1 - y0) * tf.log(1 - y_out))

    train = tf.train.GradientDescentOptimizer(1.0).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        progress = tqdm(range(1000))
        for step in progress:
            progress.set_description('loss: %8.6f, cost %8.6f' % (sess.run(loss), sess.run(cost)))
            sess.run(train)

        results = sess.run([m1, b1, m2, b2, y_out, loss])
        labels  = "m1, b1, m2, b2, y_out, loss".split(",")
        for label,result in zip(*(labels,results)) :
            print("")
            print(label)
            print(result)
