#!/usr/bin/env python3
import numpy as np
import pickle
import gzip
import tensorflow as tf


def random_tensor(*shape):
    scale = 1 / shape[-1]
    return tf.Variable(tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32))


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    hit, miss = 0, 0
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('mnist.ckpt.meta')
        saver.restore(sess, 'mnist.ckpt')
        h = tf.get_collection('h')
        predict_op = tf.get_collection('predict_op')
        vector = sess.run(h, feed_dict={'X:0': testing[0]})[0]
        prediction = sess.run(predict_op, feed_dict={'X:0': testing[0]})[0]
        print(vector)
        print(np.sum(vector, axis=-1))
        print(testing[1])
        print(prediction)
        print(np.average(prediction != testing[1]))
