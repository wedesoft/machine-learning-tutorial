#!/usr/bin/env python3
# http://www.gutenberg.org/ebooks/100.txt.utf-8
import pytest
from numpy.testing import assert_array_equal


class CharVec:
    def __init__(self, text):
        # https://stackoverflow.com/questions/13902805/list-of-all-unique-characters-in-a-string
        self.chars = ''.join(sorted(set(text)))

    def vector(self, c):
        i = self.chars.index(c)
        return [0] * i + [1] + [0] * (len(self.chars) - i - 1)

    def __call__(self, x):
        return [self.vector(c) for c in x]


class TestCharVec:
    def test_single_character(self):
        assert_array_equal(CharVec('a')('a'), [[1]])

    def test_first_character(self):
        assert_array_equal(CharVec('ab')('a'), [[1, 0]])

    def test_second_character(self):
        assert_array_equal(CharVec('ab')('b'), [[0, 1]])

    def test_multiple(self):
        assert_array_equal(CharVec('abx')('ax'), [[1, 0, 0], [0, 0, 1]])

    def test_remove_duplicates_from_ground_set(self):
        assert_array_equal(CharVec('aab')('b'), [[0, 1]])


if __name__ == '__main__':
    with open('shakespeare.txt', 'r') as f:
        data = f.read()[1:]
