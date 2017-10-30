#!/usr/bin/env python3
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
import math
import pickle
import gzip
from functools import reduce
from operator import add
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2


def random_choice(count, size):
    return np.random.choice(count, size, replace=False)


def random_selection(size, *arrays):
    indices = random_choice(len(arrays[0]), size)
    result = tuple(np.take(array, indices, axis=0) for array in arrays)
    return result[0] if len(result) == 1 else result


def show(title, img, wait=True):
    cv2.imshow(title, cv2.resize(img.reshape(28, 28), (280, 280)))
    return cv2.waitKey(-1 if wait else 1) != 27


if __name__ == '__main__':
    # http://deeplearning.net/data/mnist/mnist.pkl.gz
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')

    n_iterations = 5000
    batch_size = 256
    n_hidden1 = 300
    n_hidden2 = 150
    alpha = 100.0
    #beta = 0.01
    #rho_update = 0.01
    #rho_target = 0.2

    x = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
    m1 = tf.Variable(tf.random_normal([28 * 28, n_hidden1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([n_hidden1]))
    m2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([n_hidden2]))
    m3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([n_hidden1]))
    m4 = tf.Variable(tf.random_normal([n_hidden1, 28 * 28], stddev=0.1))
    b4 = tf.Variable(tf.random_normal([28 * 28]))
    #rho = tf.Variable(tf.constant(rho_target, shape=[n_hidden2]))
    theta = [m1, b1, m2, b2, m3, b3, m4, b4]

    a0 = x
    z1 = tf.add(tf.matmul( x, m1), b1)
    a1 = tf.sigmoid(z1)
    z2 = tf.add(tf.matmul(a1, m2), b2)
    a2 = tf.sigmoid(z2)
    z3 = tf.add(tf.matmul(a2, m3), b3)
    a3 = tf.sigmoid(z3)
    z4 = tf.add(tf.matmul(a3, m4), b4)
    a4 = tf.sigmoid(z4)
    h = a4

    #m = tf.to_float(tf.shape(x)[0])
    cost = tf.reduce_mean(tf.pow(x - h, 2))
    dtheta = tf.gradients(cost, theta)
    step = [tf.assign(value, tf.subtract(value, tf.multiply(alpha, dvalue))) for value, dvalue in zip(theta, dtheta)]
    #steps += [tf.assign(rho, tf.add(tf.multiply((1 - rho_update), rho), tf.multiply(rho_update, tf.reduce_mean(a2, 0)))),
    #         tf.assign(b2, tf.subtract(b2, tf.multiply(alpha * beta, tf.subtract(rho, rho_target))))]

    saver = tf.train.Saver()
    with tf.Session() as session:
        train = {x: training[0]}
        j_train = 0.5
        session.run(tf.global_variables_initializer())
        progress = tqdm(range(n_iterations))
        for i in progress:
            selection = random_selection(batch_size, train[x])
            mini_batch = {x: selection}
            j_train = 0.99 * j_train + 0.01 * session.run(cost, feed_dict=mini_batch)
            progress.set_description('cost: %8.6f' % j_train)
            if i % 50 == 0:
                show('original', selection[0:1], False)
                show('reconstruction', session.run(h, feed_dict={x: selection[0:1]}), False)
            #activation = session.run(tf.reduce_mean(rho), feed_dict=mini_batch)
            #progress.set_description('cost: %8.6f, rho: %8.6f' % (j_train, activation))
            session.run(step, feed_dict=mini_batch)
        tf.add_to_collection('prediction', h)
        saver.save(session, 'auto')
