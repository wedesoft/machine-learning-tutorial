#!/usr/bin/env python3
import cv2
import pickle
import gzip
import tensorflow as tf


def random_tensor(*shape):
    scale = 1 / shape[-1]
    return tf.Variable(tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32))


if __name__ == '__main__':
    # https://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    training, validation, testing = pickle.load(gzip.open('mnist.pkl.gz', 'rb'), encoding='iso-8859-1')
    print(testing[0][0].shape)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('mnist.ckpt.meta')
        saver.restore(sess, 'mnist.ckpt')
        predict_op = tf.get_collection('predict_op')
        for x, y in zip(*testing):
            cv2.imshow('x', cv2.resize(x.reshape(28, 28), 280, 280))
            prediction = sess.run(predict_op, feed_dict={'X:0': [x]})[0]
            print("prediction = %d (label = %d)" % (prediction, y))
            if cv2.waitKey(-1) == 27:
                break
