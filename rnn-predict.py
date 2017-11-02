#!/usr/bin/env python3
import sys
import numpy as np
import tensorflow as tf


class CharVec:
    def __init__(self, text):
        self.chars = np.array([ord(c) for c in  sorted(set(text))])

    def vector(self, c):
        return np.where(self.chars == ord(c), 1, 0).astype(np.float32)

    def index(self, arr):
        return ''.join([chr(self.chars[i]) for i in arr])

    def __call__(self, x):
        return np.array([self.vector(c) for c in x])

    def __len__(self):
        return len(self.chars)


def source_code():
    with open('shakespeare.txt', 'r') as f:
        return f.read()


if __name__ == '__main__':
    txt = source_code()
    char_vec = CharVec(txt)
    count = 4000
    conservative = 1.5
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('rnn.meta')
        saver.restore(sess, 'rnn')
        prob = tf.get_collection('prob')[0]
        hnext = tf.get_collection('hnext')[0]
        state = np.full(hnext.shape, 0.0, dtype=np.float32)
        output = txt[0]
        for i in range(count):
            sys.stdout.write(output)
            context = feed_dict={'x:0': char_vec(output), 'h:0': state}
            prediction = sess.run(prob, feed_dict=context) ** conservative
            idx = np.argwhere(np.cumsum(prediction) >= np.sum(prediction) * np.random.rand())[0]
            output = char_vec.index([idx])
            state = sess.run(hnext, feed_dict=context)
