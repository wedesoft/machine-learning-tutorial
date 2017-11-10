#!/usr/bin/env python3
# http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
from functools import reduce
from operator import add
import pickle
import gzip
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Scale:
    def __init__(self, features, max_scale=10.0):
        self.average = np.average(features, axis=0)
        self.deviation = np.maximum(np.std(features, axis=0), 1.0 / max_scale)

    def __call__(self, values):
        return np.subtract(values, self.average) / self.deviation


def multi_class_label(labels, num_classes):
    index = np.arange(len(labels) * num_classes).reshape(len(labels), num_classes)
    return np.where(np.equal(index % num_classes, np.expand_dims(labels, -1)), 1, 0)


def random_choice(count, size):
    return np.random.choice(count, size, replace=False)


def random_selection(size, *arrays):
    indices = random_choice(len(arrays[0]), size)
    result = tuple(np.take(array, indices, axis=0) for array in arrays)
    return result[0] if len(result) == 1 else result


if __name__ == '__main__':
	# http://deeplearning.net/data/mnist/mnist.pkl.gz
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    n_classes = 10
    n_iterations = 100000
    batch_size = 100
    alpha = 0.2
    regularize = 0.016
    scale = Scale(training[0], 1000.0)

    x = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    c1 = tf.Variable(tf.truncated_normal([3, 3, 1, 4], stddev=1.0/9))
    c2 = tf.Variable(tf.truncated_normal([3, 3, 4, 16], stddev=1.0/9))
    m3 = tf.Variable(tf.truncated_normal([400, 64], stddev=1.0/400))
    b3 = tf.Variable(tf.constant(1.0/16, shape=[64]))
    m4 = tf.Variable(tf.truncated_normal([64, 10], stddev=1.0/64))
    b4 = tf.Variable(tf.constant(1.0/10, shape=[10]))
    theta = [c1, c2, m3, b3, m4, b4]
    reg_candidates = [c1, c2, m3, m4]

    a0 = tf.reshape(x, [-1, 28, 28, 1])
    z1 = tf.nn.conv2d(a0, c1, strides=[1, 1, 1, 1], padding='VALID')
    a1 = tf.nn.softplus(tf.nn.max_pool(z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
    z2 = tf.nn.conv2d(a1, c2, strides=[1, 1, 1, 1], padding='VALID')
    a2 = tf.nn.softplus(tf.nn.max_pool(z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
    a2_flat = tf.sigmoid(tf.reshape(a2, [-1, 400]))
    z3 = tf.add(tf.matmul(a2_flat, m3), b3)
    a3 = tf.sigmoid(z3)
    z4 = tf.add(tf.matmul(a3, m4), b4)
    a4 = tf.sigmoid(z4)
    h = a4

    prediction = tf.argmax(h, axis=-1)

    m = tf.cast(tf.size(y) / n_classes, tf.float32)
    reg_term = reduce(add, [tf.reduce_sum(tf.square(parameter)) for parameter in reg_candidates]) / (m * 2)
    safe_log = lambda v: tf.log(tf.clip_by_value(v, 1e-10, 1.0))
    error_term = -tf.reduce_sum(y * safe_log(h) + (1 - y) * safe_log(1 - h)) / m
    cost = error_term + regularize * reg_term
    dtheta = tf.gradients(cost, theta)
    step = [tf.assign(value, tf.subtract(value, tf.multiply(alpha, dvalue))) for value, dvalue in zip(theta, dtheta)]
    rmsd = tf.reduce_sum(tf.square(h - y)) / (2 * m)

    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    progress = tqdm(range(n_iterations))
    train = {x: scale(training[0]), y: multi_class_label(training[1], n_classes)}
    validate = {x: scale(validation[0] ), y: multi_class_label(validation[1], n_classes)}
    j_train = 0.5
    for i in progress:
        selection = random_selection(batch_size, train[x], train[y])
        mini_batch = {x: selection[0], y: selection[1]}
        j_train = 0.99 * j_train + 0.01 * session.run(rmsd, feed_dict=mini_batch)
        progress.set_description('%8.6f' % j_train)
        session.run(step, feed_dict=mini_batch)
    output = session.run(prediction, feed_dict=validate)
    print('validation labels:', validation[1])
    print('predictions      :', output)
    print('validation error rate:', np.average(output != validation[1]))
    tf.add_to_collection('prediction', prediction)
    saver.save(session, './model')
