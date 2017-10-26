#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import cv2


class Convolution:
    def __init__(self, image_shape, kernel_shape):
        self.x = tf.placeholder(tf.float32, image_shape, name='x')
        self.kernel = tf.placeholder(tf.float32, kernel_shape + (1, 1), name='kernel')
        image = tf.reshape(self.x, (-1,) + image_shape + (1,))
        self.fun = tf.nn.conv2d(image, self.kernel, strides=(1, 1, 1, 1), padding='SAME')

    def __call__(self, image, kernel):
        with tf.Session() as sess:
            kernel = np.reshape(kernel, self.kernel.shape)
            result = sess.run(self.fun, feed_dict={self.x: image, self.kernel: kernel})
            return result.reshape(image.shape)


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    convolution = None
    while True:
        _, frame = camera.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if convolution is None:
            convolution = Convolution(grey.shape, (7, 7))
        cv2.imshow('convolution', convolution(grey, np.full((7, 7), 1/49)).astype(np.uint8))
        if cv2.waitKey(1) == 27:
            break
