#!/usr/bin/env python3
import pickle
import gzip
import tensorflow as tf


training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
x = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
y = tf.placeholder(tf.float32, [None, 10])
c1 = tf.Variable(tf.truncated_normal([3, 3, 1, 4], stddev=1.0/9))
b1 = tf.Variable(tf.constant(1.0/4, shape=[4]))
c2 = tf.Variable(tf.truncated_normal([3, 3, 4, 16], stddev=1.0/9))
b2 = tf.Variable(tf.constant(1.0/16, shape=[16]))
m3 = tf.Variable(tf.truncated_normal([400, 64], stddev=1.0/400))
b3 = tf.Variable(tf.constant(1.0/16, shape=[64]))
m4 = tf.Variable(tf.truncated_normal([64, 10], stddev=1.0/64))
b4 = tf.Variable(tf.constant(1.0/10, shape=[10]))
theta = [c1, b1, c2, b2, m3, b3, m4, b4]

a0 = tf.reshape(x, [-1, 28, 28, 1])
z1 = tf.add(tf.nn.conv2d(a0, c1, strides=[1, 1, 1, 1], padding='VALID'), b1)
a1 = tf.nn.max_pool(z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
z2 = tf.add(tf.nn.conv2d(a1, c2, strides=[1, 1, 1, 1], padding='VALID'), b2)
a2 = tf.nn.max_pool(z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
a2_flat = tf.sigmoid(tf.reshape(a2, [-1, 400]))
z3 = tf.add(tf.matmul(a2_flat, m3), b3)
a3 = tf.sigmoid(z3)
z4 = tf.add(tf.matmul(a3, m4), b4)
a4 = tf.sigmoid(z4)
y = a4

m = tf.cast(tf.size(y) / n_classes, tf.float32)


session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
session.run(a2_flat, feed_dict={x: training[0][0:1]})
