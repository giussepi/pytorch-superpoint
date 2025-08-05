# -*- coding: utf-8 -*-
""" utils/test/test_utils.py """

import unittest

import numpy as np


from utils.utils import nms_fast


class Test_nms_fast(unittest.TestCase):

    def setUp(self):
        self.input_corners = np.array([
            [0., 10., 12., 14., 59., 54.,  0.,  59.],  # x
            [0., 10., 13., 14., 29., 25., 29.,  0.],  # y
            [.9,  .6,  .8,  .5,  .6, .55,   .4,  .7],  # confidence score
        ])
        self.height = 30
        self.width = 60

    def tearDown(self):
        del self.input_corners

    def test_1(self):
        """ Tests radius = 4 """
        radius = 4
        expected_corners = np.array([
            [0., 12., 59., 59., 54., 0.],
            [0., 13.,  0., 29., 25., 29.],
            [.9, .8,   .7, .6,  .55, .4],
        ])
        expected_idxs = np.array([0, 2, 7, 4, 5, 6])
        actual_corners, actual_idxs = nms_fast(self.input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_2(self):
        """ Tests radius = 5"""
        radius = 5
        expected_corners = np.array([
            [0., 12., 59., 59., 0.],
            [0., 13.,  0., 29., 29.],
            [.9, .8,   .7, .6,  .4],
        ])
        expected_idxs = np.array([0, 2, 7, 4, 6])
        actual_corners, actual_idxs = nms_fast(self.input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_3(self):
        """ Tests radius = 0 """
        radius = 0
        expected_corners = np.array([
            [0., 12., 59., 10., 59.,  54., 14.,  0.],
            [0., 13.,  0., 10., 29.,  25., 14., 29.],
            [.9, .8,   .7, .6,   .6,  .55, .5,   .4],
        ])
        expected_idxs = np.array([0, 2, 7, 1, 4, 5, 3, 6])
        actual_corners, actual_idxs = nms_fast(self.input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_4(self):
        """ Tests radius covering the whole image """
        radius = 60
        expected_corners = np.array([
            [0.],
            [0.],
            [.9],
        ])
        expected_idxs = np.array([0])
        actual_corners, actual_idxs = nms_fast(self.input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_5(self):
        """ Tests points with the same confidence scores """
        radius = 4
        input_corners = np.array([
            [0., 3.,  10., 12., 14., 59., 54.,  0., 55., 59.],  # x
            [0., 3.,  10., 13., 14., 29., 25., 29., 4.,  0.],  # y
            [.9, .9,   .6,  .8,  .5,  .6, .55,  .4, .7, .7],  # confidence score
            # 0   1    2    3    4    5    6     7   8   9
        ])

        expected_corners = np.array([
            [0., 3., 12., 55., 59., 59., 54.,  0.],
            [0., 3., 13.,  4.,  0., 29., 25., 29.],
            [.9, .9,  .8,  .7,  .7,  .6, .55,  .4],
        ])
        expected_idxs = np.array([0, 1, 3, 8, 9, 5, 6, 7])
        actual_corners, actual_idxs = nms_fast(input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_6(self):
        """ Tests single point input """
        radius = 4
        input_corners = np.array([
            [0.],  # x
            [0.],  # y
            [.9],  # confidence score
        ])

        expected_corners = input_corners
        expected_idxs = np.array([0])
        actual_corners, actual_idxs = nms_fast(input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_7(self):
        """ Tests empty input """
        radius = 4
        input_corners = np.empty((3, 0))
        expected_corners = input_corners
        expected_idxs = np.empty(0)
        actual_corners, actual_idxs = nms_fast(input_corners, self.height, self.width, radius)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))


if __name__ == "__main__":
    unittest.main()
