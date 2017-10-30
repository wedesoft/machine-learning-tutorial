#!/usr/bin/env python3
# http://blog.aloni.org/posts/backprop-with-tensorflow/
import random
import numpy as np
import tensorflow as tf
import pickle
import gzip
import cv2


def show(title, img, wait=True):
    cv2.imshow(title, cv2.resize(img.reshape(28, 28), (280, 280)))
    return cv2.waitKey(-1 if wait else 1) != 27


if __name__ == '__main__':
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('auto.meta')
        saver.restore(sess, 'auto')
        image = np.random.rand(1, 784)
        prediction = tf.get_collection('prediction')[0]
        while show('prediction', image, False):
            index = random.randrange(len(testing[0]))
            image = testing[0][index:index + 1]
            show('input', image, False)
            image = sess.run(prediction, feed_dict={'x:0': image})
