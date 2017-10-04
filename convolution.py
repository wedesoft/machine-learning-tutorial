#!/usr/bin/env python3
from numpy.testing import assert_array_equal
import theano
import theano.tensor as T
import numpy as np
import cv2


def to_grey(input_image):
    input_image = np.uint8(input_image)
    if len(input_image.shape) == 2:
        return input_image
    else:
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)


class TestToGrey:
    def test_returns_2d_array(self):
        assert to_grey(np.zeros((320,240,3))).shape == (320, 240)

    def test_different_image_size(self):
        assert to_grey(np.zeros((640,480,3))).shape == (640, 480)

    def test_check_content(self):
        assert_array_equal(to_grey([[[100,100,100]]]), [[100]])

    def test_check_red_channel(self):
        assert_array_equal(to_grey([[[0, 0, 100]]]), [[30]])

    def test_check_green_channel(self):
        assert_array_equal(to_grey([[[0, 100, 0]]]), [[59]])

    def test_check_blue_channel(self):
        assert_array_equal(to_grey([[[100, 0, 0]]]), [[11]])

    def test_grey_input(self):
        assert to_grey(np. zeros((320,240))).shape == (320, 240)


class Convolution:
    def __init__(self, image_size, kernel_size):
        x = T.tensor4()
        y = T.tensor4()
        self.image_shape = (1, 1, image_size, 1)
        self.filter_shape = (1, 1, kernel_size, 1)
        self.start = kernel_size // 2
        self.end = image_size + kernel_size // 2
        expression = T.nnet.conv.conv2d(x, y, image_shape=self.image_shape, filter_shape=self.filter_shape, border_mode='full')
        self.fun = theano.function((x, y), outputs=expression)

    def __call__(self, image, kernel):
        image = np.array(image)
        kernel = np.array(kernel)
        result = self.fun(image.reshape(self.image_shape), kernel.reshape(self.filter_shape))
        return result[0, 0, self.start:self.end, 0]


class TestConvolution:
    def test_trivial_1d(self):
        assert_array_equal(Convolution(5, 1)([2, 3, 5, 7, 11], [1]), [2, 3, 5, 7, 11])

    def test_box_1d(self):
        assert_array_equal(Convolution(5, 3)([0, 0, 1, 0, 0], [1, 1, 1]), [0, 1, 1, 1, 0])


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        grey = to_grey(frame)
        cv2.imshow('convolution', grey)
        if cv2.waitKey(1) == 27:
            break
