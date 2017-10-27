#!/usr/bin/env python3
# https://gist.github.com/Cospel/f364df97b4944cec2dc0
import math
import pickle
import gzip
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm


def sample(probability):
    return tf.nn.relu(tf.sign(probability - tf.random_uniform(tf.shape(probability))))


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
    epsilon = 1.0
    n_hidden = 200
    batch_size = 100
    n_iterations = 50000

    v = tf.placeholder(tf.float32, [None, 28 * 28], name='v')
    w = tf.Variable(tf.truncated_normal([28 * 28, n_hidden], stddev=1.0/math.sqrt(784 * n_hidden)))
    a = tf.Variable(tf.constant(0.5, shape=[784]))
    b = tf.Variable(tf.constant(0.5, shape=[n_hidden]))
    theta = [w, a, b]

    m = tf.to_float(tf.shape(v)[0])
    hp = tf.sigmoid(tf.add(tf.matmul(v, w), b))
    h = sample(hp)
    positive_gradient = tf.matmul(tf.transpose(v), h)
    vp = tf.sigmoid(tf.add(tf.matmul(h, tf.transpose(w)), a))
    vs = sample(vp)
    hs = sample(tf.sigmoid(tf.add(tf.matmul(vs, w), b)))
    negative_gradient = tf.matmul(tf.transpose(vs), hs)

    error = tf.reduce_mean(tf.square(v - vs))
    dw = tf.subtract(positive_gradient, negative_gradient) / m
    da = tf.reduce_mean(tf.subtract(v, vs), 0)
    db = tf.reduce_mean(tf.subtract(h, hs), 0)
    dtheta = [dw, da, db]
    step = [tf.assign(value, tf.add(value, tf.multiply(epsilon, dvalue))) for value, dvalue in zip(theta, dtheta)]

    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    progress = tqdm(range(n_iterations))
    train = training[0]
    j_train = 0.5
    for i in progress:
        selection = random_selection(batch_size, train)
        mini_batch={v: selection}
        j_train = 0.99 * j_train + 0.01 * session.run(error, feed_dict=mini_batch)
        progress.set_description('%8.6f' % j_train)
        if i % 50 == 0:
            show('original', selection[0:1], False)
            show('reconstruction', session.run(vp, feed_dict={v: selection[0:1]}), False)
        session.run(step, feed_dict=mini_batch)
    tf.add_to_collection('vs', vs)
    tf.add_to_collection('vp', vp)
    saver.save(session, 'rbm')
