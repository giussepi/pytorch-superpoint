# -*- coding: utf-8 -*-
""" utils/test/test_utils.py """

import unittest

import numpy as np


from utils.utils import nms_fast


class Test_nms_fast(unittest.TestCase):

    def setUp(self):
        self.in_corners = np.array([
            [0., 10., 12., 14., 29., 24.,  0.,  29.],  # x
            [0., 10., 13., 14., 29., 25., 29.,  0.],  # y
            [.9,  .6,  .8,  .5,  .6, .55,   .4,  .7],  # confidence score
        ])
        self.height = 30
        self.width = 30

    def tearDown(self):
        del self.in_corners

    def test_1(self):
        dist_thresh = 4
        expected_corners = np.array([
            [0., 12., 29., 29., 24., 0.],
            [0., 13.,  0., 29., 25., 29.],
            [.9, .8,   .7, .6,  .55, .4],
        ])
        expected_idxs = np.array([0, 2, 7, 4, 5, 6])
        actual_corners, actual_idxs = nms_fast(self.in_corners, self.height, self.width, dist_thresh)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_2(self):
        dist_thresh = 5
        expected_corners = np.array([
            [0., 12., 29., 29., 0.],
            [0., 13.,  0., 29., 29.],
            [.9, .8,   .7, .6,  .4],
        ])
        expected_idxs = np.array([0, 2, 7, 4, 6])
        actual_corners, actual_idxs = nms_fast(self.in_corners, self.height, self.width, dist_thresh)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_3(self):
        dist_thresh = 0
        expected_corners = np.array([
            [0., 12., 29., 10., 29.,  24., 14.,  0.],
            [0., 13.,  0., 10., 29.,  25., 14., 29.],
            [.9, .8,   .7, .6,   .6,  .55, .5,   .4],
        ])
        expected_idxs = np.array([0, 2, 7, 1, 4, 5, 3, 6])
        actual_corners, actual_idxs = nms_fast(self.in_corners, self.height, self.width, dist_thresh)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))

    def test_4(self):
        dist_thresh = 30
        expected_corners = np.array([
            [0.],
            [0.],
            [.9],
        ])
        expected_idxs = np.array([0])
        actual_corners, actual_idxs = nms_fast(self.in_corners, self.height, self.width, dist_thresh)

        self.assertTrue(np.array_equal(actual_corners, expected_corners))
        self.assertTrue(np.array_equal(actual_idxs, expected_idxs))


if __name__ == "__main__":
    unittest.main()
