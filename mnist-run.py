#!/usr/bin/env python3
import numpy as np
import pickle
import gzip
import tensorflow as tf


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    hit, miss = 0, 0
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('mnist.ckpt.meta')
        saver.restore(sess, 'mnist.ckpt')
        prediction = tf.get_collection('prediction')
        prediction = sess.run(prediction, feed_dict={'x:0': testing[0]})[0]
        print('test labels:', testing[1])
        print('predictions:', prediction)
        print('error rate:', np.average(prediction != testing[1]))
