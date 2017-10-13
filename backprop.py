#!/usr/bin/env python3
# https://stackoverflow.com/questions/44210561/how-do-backpropagation-works-in-tensorflow
# https://github.com/craffel/theano-tutorial/blob/master/Backpropagation.ipynb
import pytest
import math
from numpy.testing import assert_array_equal
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    ### constant data
    x  = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_ = [[0], [1], [1], [0]]

    ### induction
    # 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

    # Layer 0 = the x2 inputs
    x0 = tf.constant(x , dtype=tf.float32)
    y0 = tf.constant(y_, dtype=tf.float32)

    # Layer 1 = the 2x2 hidden sigmoid
    m1 = tf.Variable(tf.random_uniform([2, 2], minval=0.1, maxval=0.9, dtype=tf.float32))
    b1 = tf.Variable(tf.random_uniform([2]   , minval=0.1, maxval=0.9, dtype=tf.float32))
    h1 = tf.sigmoid(tf.matmul(x0, m1) + b1)

    # Layer 2 = the 2x1 sigmoid output
    m2 = tf.Variable(tf.random_uniform([2, 1], minval=0.1, maxval=0.9, dtype=tf.float32))
    b2 = tf.Variable(tf.random_uniform([1]   , minval=0.1, maxval=0.9, dtype=tf.float32))
    y_out = tf.sigmoid(tf.matmul(h1, m2) + b2)

    ### loss
    # loss : sum of the squares of y0 - y_out
    loss = tf.reduce_sum(tf.square(y0 - y_out))

    # training step : gradient decent (1.0) to minimize loss
    train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    ### training
    # run 500 times using all the X and Y
    # print out the loss and any other interesting info
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for step in range(500) :
        sess.run(train)

      results = sess.run([m1, b1, m2, b2, y_out, loss])
      labels  = "m1, b1, m2, b2, y_out, loss".split(",")
      for label,result in zip(*(labels,results)) :
        print("")
        print(label)
        print(result)
