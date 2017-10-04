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
    def __init__(self, image_shape, kernel_shape):
        x = T.dmatrix()
        y = T.dmatrix()
        expression = T.nnet.conv.conv2d(x, y, (1, image_shape), (1, kernel_shape))

    def __call__(self, image, kernel):
        return image


class TestConvolution:
    def test_trivial_1d(self):
        assert_array_equal(Convolution(5, 1)([2, 3, 5, 7, 11], [1]), [2, 3, 5, 7, 11])

    def test_box_1d(self):
        assert_array_equal(Convolution(5, 1)([0, 0, 1, 0, 0], [1, 1, 1]), [0, 1, 3, 1, 0])


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        grey = to_grey(frame)
        cv2.imshow('convolution', grey)
        if cv2.waitKey(1) == 27:
            break
