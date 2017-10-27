#!/usr/bin/env python3
# http://blog.aloni.org/posts/backprop-with-tensorflow/
import random
import pickle
import gzip
import numpy as np
import tensorflow as tf
import cv2


def show(title, img, wait=True):
    cv2.imshow(title, cv2.resize(img.reshape(28, 28), (280, 280)))
    return cv2.waitKey(-1 if wait else 1) != 27


if __name__ == '__main__':
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('rbm.meta')
        saver.restore(sess, 'rbm')
        image = np.random.rand(1, 784)
        vp = tf.get_collection('vp')[0]
        i = 0
        while show('dream', image, False):
            if i % 500 == 0:
                index = random.randrange(len(testing[0]))
                image = testing[0][index:index + 1]
                show('input', image, False)
            else:
                image = sess.run(vp, feed_dict={'v:0': np.where(image >= np.random.rand(1, 784), 1, 0)})
            i += 1
