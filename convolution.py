#!/usr/bin/env python3
from numpy.testing import assert_array_equal
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
        assert to_grey(np.zeros((320,240))).shape == (320, 240)


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        grey = to_grey(frame)
        cv2.imshow('convolution', grey)
        if cv2.waitKey(1) == 27:
            break
