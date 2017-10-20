#!/usr/bin/env python3
# http://blog.aloni.org/posts/backprop-with-tensorflow/
import numpy as np
import tensorflow as tf
import pickle
import gzip


class Scale:
    def __init__(self, features, max_scale=10.0):
        self.average = np.average(features, axis=0)
        self.deviation = np.maximum(np.std(features, axis=0), 1 / max_scale)

    def __call__(self, values):
        return np.subtract(values, self.average) / self.deviation


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    scale = Scale(training[0], 1000)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model.meta')
        saver.restore(sess, 'model')
        prediction = tf.get_collection('prediction')[0]
        output = sess.run(prediction, feed_dict={'x:0': scale(testing[0])}) 
        print('test labels:', testing[1])
        print('predictions:', output)
        print('error rate:', np.average(output != testing[1]))
