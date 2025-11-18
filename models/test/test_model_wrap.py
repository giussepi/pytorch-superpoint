# -*- coding: utf-8 -*-
""" models/test/test_model_wrap.py """

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from models.model_wrap import PointTracker2


class Test_PointTracker2(unittest.TestCase):

    def setUp(self):
        self.num_frames = 3
        self.desc_threshold = .75
        self.tracker = PointTracker2(self.num_frames, self.desc_threshold)
        self.tracker.tracks = np.stack([
            np.arange(5),  # id
            np.array([.33, .15, .07, .55, .87]),  # cumulative_avg
            np.array([2, 11, 4, 6, 9]),  # num_desc
            np.arange(5),  # pt id 0
            np.arange(5, 10),  # pt id 1
            np.arange(10, 15),  # pt id 2
        ], axis=1)
        self.tracker.total_tracks = 5

    def tearDown(self):
        del self.tracker

    def test_1_1(self):
        """ Tests delete_oldest_points: standard case """
        self.tracker.delete_oldest_points()
        self.assertEqual(self.tracker.tracks.shape, (5, 5))
        self.assertTrue(np.array_equal(self.tracker.tracks[:, 3], np.arange(5, 10)))
        self.assertTrue(np.array_equal(self.tracker.tracks[:, 4], np.arange(10, 15)))

    def test_1_2(self):
        """ Tests delete_oldest_points: case no tracks column """
        self.tracker.tracks = self.tracker.tracks[:, 3:]
        expected_shape = (5, 3)
        expected_tracks = self.tracker.tracks.copy()
        self.tracker.delete_oldest_points()
        self.assertEqual(self.tracker.tracks.shape, expected_shape)
        np.testing.assert_array_equal(self.tracker.tracks, expected_tracks)

    def test_1_3(self):
        """ Tests delete_oldest_points: case empty tracks """
        self.tracker.tracks = np.empty((0, self.num_frames + 3))
        expected_shape = (0, 5)
        expected_tracks = np.empty((0, self.num_frames + 2))
        self.tracker.delete_oldest_points()
        self.assertEqual(self.tracker.tracks.shape, expected_shape)
        np.testing.assert_array_equal(self.tracker.tracks, expected_tracks)

    @patch('models.model_wrap.PointTracker2.update_ca_scores')
    def test_2_1(self, mocked_update_ca_scores):
        """ Tests append_matched_tracks: valid new idxs """
        matches = np.array([
            [13, 11, 10],
            [22, 21, 24],
        ])
        new_scores = np.random.rand(3)
        # Verifying the idxs are appended as the new last column
        self.tracker.append_matched_tracks(matches, new_scores)
        self.assertEqual(self.tracker.tracks.shape, (5, 7))
        self.assertTrue(np.array_equal(self.tracker.tracks[:, -1], np.array([24, 21, -1, 22, -1])))
        # Verifying update_ca_scores is called. This method is already tested so no need to do it again
        mocked_update_ca_scores.assert_called_once_with(new_scores, [3, 1, 0])

    @patch('models.model_wrap.PointTracker2.update_ca_scores')
    def test_2_2(self, mocked_update_ca_scores):
        """ Tests append_matched_tracks: idxs full of -1 """
        matches = np.full((2, 5), -1)
        new_scores = np.zeros(5)
        # Verifying the idxs are appended as the new last column
        self.tracker.append_matched_tracks(matches, new_scores)
        self.assertEqual(self.tracker.tracks.shape, (5, 7))
        np.testing.assert_array_equal(self.tracker.tracks[:, -1], matches[1])
        # Verifying update_ca_scores is not called.
        mocked_update_ca_scores.assert_not_called()

    @patch('models.model_wrap.PointTracker2.update_ca_scores')
    def test_2_3(self, mocked_update_ca_scores):
        """ Tests append_matched_tracks: empty tracks and empty tracks matrix  """
        # NOTE: this case only happens when the tracks matrix is empty
        self.tracker = PointTracker2(self.num_frames, self.desc_threshold)
        matches = np.empty((2, 0))
        new_scores = np.zeros(0)
        # Verifying the idxs are appended as the new last column
        self.tracker.append_matched_tracks(matches, new_scores)
        self.assertEqual(self.tracker.tracks.shape, (0, 7))
        np.testing.assert_array_equal(self.tracker.tracks[:, -1], matches[1])
        # Verifying update_ca_scores is not called.
        mocked_update_ca_scores.assert_not_called()

    def test_3_1(self):
        """ Tests update_ca_scores: standard case """
        new_ca_scores = np.array([.97, .22, .66])
        idxs_to_update = [4, 0, 1]
        expected_shape = self.tracker.tracks.shape
        old_num_desc = self.tracker.tracks[:, 2].copy()
        expected_ca = self.tracker.tracks[:, 1].copy()
        expected_ca[idxs_to_update] = self.tracker.compute_cumulative_avg(
            new_ca_scores, np.array([9, 2, 11])+1, np.array([.87, .33, .15])
        )
        expected_num_desc = old_num_desc.copy()
        expected_num_desc[idxs_to_update] += 1
        self.tracker.update_ca_scores(new_ca_scores, idxs_to_update)

        # verifying tracks shapes
        self.assertEqual(self.tracker.tracks.shape, expected_shape)
        # verifying cumulative_avg
        np.testing.assert_array_equal(self.tracker.tracks[:, 1], expected_ca)
        # verifying num_desc
        self.assertFalse(np.array_equal(self.tracker.tracks[:, 2], old_num_desc))
        np.testing.assert_array_equal(self.tracker.tracks[:, 2], expected_num_desc)

    def test_3_2(self):
        """ Tests update_ca_scores: case empty new_scores """
        new_ca_scores = np.empty([])
        idxs_to_update = []

        with self.assertRaises(AssertionError):
            self.tracker.update_ca_scores(new_ca_scores, idxs_to_update)

    def test_4_1(self):
        """ Tests append_unmatched_tracks: case no tracks provided """
        unmatched_tracks = None
        expected_tracks = self.tracker.tracks.copy()
        self.tracker.append_unmatched_tracks(unmatched_tracks)
        actual_tracks = self.tracker.tracks
        self.assertTrue(np.array_equal(actual_tracks, expected_tracks))

    def test_4_2(self):
        """ Tests append_unmatched_tracks: updating non-empty tracks matrix """
        unmatched_tracks = np.array([3, 0, 2, 1])
        expected_tracks = np.vstack([
            self.tracker.tracks,
            [
                [5, 0, 0, -1, -1, 3],
                [6, 0, 0, -1, -1, 0],
                [7, 0, 0, -1, -1, 2],
                [8, 0, 0, -1, -1, 1],
            ]
        ])
        self.tracker.append_unmatched_tracks(unmatched_tracks)
        actual_tracks = self.tracker.tracks
        self.assertTrue(np.array_equal(actual_tracks, expected_tracks))

    def test_4_3(self):
        """ Tests append_unmatched_tracks: updating after some deletions """
        unmatched_tracks = np.array([3, 0, 2, 1])
        self.tracker.tracks = np.delete(self.tracker.tracks, [1, 3, 4], axis=0)
        expected_tracks = np.vstack([
            self.tracker.tracks,
            [
                [5, 0, 0, -1, -1, 3],
                [6, 0, 0, -1, -1, 0],
                [7, 0, 0, -1, -1, 2],
                [8, 0, 0, -1, -1, 1],
            ]
        ])
        self.tracker.append_unmatched_tracks(unmatched_tracks)
        actual_tracks = self.tracker.tracks
        np.testing.assert_array_equal(actual_tracks, expected_tracks)

    def test_4_4(self):
        """ Tests append_unmatched_tracks: updating empty tracks matrix """
        self.tracker = PointTracker2(self.num_frames, self.desc_threshold)
        unmatched_tracks = np.array([3, 0, 2, 1])
        expected_tracks = np.array([
            [0, 0, 0, -1, -1, 3],
            [1, 0, 0, -1, -1, 0],
            [2, 0, 0, -1, -1, 2],
            [3, 0, 0, -1, -1, 1],
        ])
        self.tracker.append_unmatched_tracks(unmatched_tracks)
        actual_tracks = self.tracker.tracks
        self.assertTrue(np.array_equal(actual_tracks, expected_tracks))

    def test_5(self):
        """ Tests compute_cumulative_avg  """
        scores = np.stack([
            np.arange(0, 10, 2),  # time = 1
            np.arange(10, 20, 2),  # time = 2
            np.arange(20, 30, 2),  # time = 3
        ], axis=1)

        # time 1
        n = np.zeros(5)  # previous_num_processed_values
        expected_values_1 = scores[:, 0]
        actual_values_1 = self.tracker.compute_cumulative_avg(scores[:, 0], n, np.zeros_like(scores[:, 0]))
        self.assertTrue(np.array_equal(actual_values_1, expected_values_1))

        # time 2
        n += 1  # previous_num_processed_values
        expected_values_2 = scores[:, :2].mean(axis=1)
        actual_values_2 = self.tracker.compute_cumulative_avg(scores[:, 1], n, actual_values_1)
        self.assertTrue(np.array_equal(actual_values_2, expected_values_2))

        # time 3
        n += 1  # previous_num_processed_values
        expected_values_3 = scores.mean(axis=1)
        actual_values_3 = self.tracker.compute_cumulative_avg(scores[:, 2], n, actual_values_2)
        self.assertTrue(np.array_equal(actual_values_3, expected_values_3))

    @patch('models.model_wrap.PointTracker2.append_unmatched_tracks')
    @patch('models.model_wrap.PointTracker2.append_matched_tracks')
    def test_6_1(self, mocked_append_matched_tracks, mocked_append_unmatched_tracks):
        """ Tests append_tracks: matched and unmatched """
        matched = np.array([
            [13, 11, 10],  # desc1 idx
            [22, 21, 24],  # desc2 idx
            [.7, .1, .33],  # scores
        ])
        unmatched = np.array([4, 0, 1, 3, 2])  # idxs
        self.tracker.append_tracks(matched=matched, unmatched=unmatched)

        mocked_append_matched_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_matched_tracks.call_args[0][0], matched[[0, 1]])
        np.testing.assert_array_equal(mocked_append_matched_tracks.call_args[0][1], matched[2])
        mocked_append_unmatched_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_unmatched_tracks.call_args[0][0], unmatched)

    @patch('models.model_wrap.PointTracker2.append_unmatched_tracks')
    @patch('models.model_wrap.PointTracker2.append_matched_tracks')
    def test_6_2(self, mocked_append_matched_tracks, mocked_append_unmatched_tracks):
        """ Tests append_tracks: only matched """
        matched = np.array([
            [13, 11, 10],  # desc1 idx
            [22, 21, 24],  # desc2 idx
            [.7, .1, .33],  # scores
        ])
        unmatched = None  # idxs
        self.tracker.append_tracks(matched=matched, unmatched=unmatched)

        mocked_append_matched_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_matched_tracks.call_args[0][0], matched[[0, 1]])
        np.testing.assert_array_equal(mocked_append_matched_tracks.call_args[0][1], matched[2])
        mocked_append_unmatched_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_unmatched_tracks.call_args[0][0], np.empty(0, dtype=int))

    @patch('models.model_wrap.PointTracker2.append_unmatched_tracks')
    @patch('models.model_wrap.PointTracker2.append_matched_tracks')
    def test_6_3(self, mocked_append_matched_tracks, mocked_append_unmatched_tracks):
        """ Tests append_tracks: only unmatched """
        matched = None
        unmatched = np.array([4, 0, 1, 3, 2])  # idxs
        self.tracker.append_tracks(matched=matched, unmatched=unmatched)

        mocked_append_matched_tracks.assert_called_once()
        np.testing.assert_array_equal(
            mocked_append_matched_tracks.call_args[0][0],
            np.full((2, self.tracker.tracks.shape[0]), -1, dtype=int)
        )
        np.testing.assert_array_equal(
            mocked_append_matched_tracks.call_args[0][1],
            np.zeros(self.tracker.tracks.shape[0], dtype=int)
        )
        mocked_append_unmatched_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_unmatched_tracks.call_args[0][0], unmatched)

    def test_6_4(self):
        """ Tests append_tracks: Neither matched nor unmatched """
        with self.assertRaises(AssertionError):
            self.tracker.append_tracks(matched=None, unmatched=None)

    def test_5_1(self):
        """ Tests delete_empty_tracks_rows: standard case """
        self.tracker.tracks[:, 3:] = np.array([
            [0,  1, -1],
            [-1, -1, -1],
            [-1, -1,  0],
            [1,  2,  3],
            [-1, -1, -1],
        ])
        expected_tracks = self.tracker.tracks[[0, 2, 3], :].copy()
        self.tracker.delete_empty_tracks_rows()
        actual_tracks = self.tracker.tracks
        np.testing.assert_array_equal(actual_tracks, expected_tracks)

    def test_5_2(self):
        """ Tests delete_empty_tracks_rows: case nothing to delete """
        self.tracker.tracks[:, 3:] = np.array([
            [0,  1, -1],
            [-1, -1, 0],
            [-1, -1, 0],
            [1,   2,  3],
            [-1, -1, 1],
        ])
        expected_tracks = self.tracker.tracks.copy()
        self.tracker.delete_empty_tracks_rows()
        actual_tracks = self.tracker.tracks
        np.testing.assert_array_equal(actual_tracks, expected_tracks)

    def test_5_3(self):
        """ Tests delete_empty_tracks_rows: case empty tracks """
        self.tracker.tracks = np.empty((0, self.num_frames + 3))
        expected_tracks = self.tracker.tracks.copy()
        self.tracker.delete_empty_tracks_rows()
        actual_tracks = self.tracker.tracks
        np.testing.assert_array_equal(actual_tracks, expected_tracks)

    def test_5_4(self):
        """ Tests delete_empty_tracks_rows: case all rows deleted """
        self.tracker.tracks[:, 3:] = np.array([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ])
        expected_tracks = np.empty((0, self.num_frames + 3))
        self.tracker.delete_empty_tracks_rows()
        actual_tracks = self.tracker.tracks
        np.testing.assert_array_equal(actual_tracks, expected_tracks)

    @patch('models.model_wrap.PointTracker2.append_tracks')
    @patch('models.model_wrap.PointTracker2.nn_match_two_way')
    @patch('models.model_wrap.PointTracker2.delete_empty_tracks_rows')
    @patch('models.model_wrap.PointTracker2.delete_oldest_points')
    def test_7_1(
            self,
            mocked_delete_oldest_points,
            mocked_delete_empty_tracks_rows,
            mocked_nn_match_two_way,
            mocked_append_tracks
    ):
        """ Tests update: case first time called """
        pts = np.array([
            [1.0, 1.0, 2.00, 2.00, 3.00],  # x_i
            [1.0, 2.0, 2.00, 1.00, 2.00],  # y_i
            [0.7, 0.8, 0.74, 0.75, 0.76],  # confidence_i
        ])
        desc = np.empty((10, 5))
        self.tracker.update(pts, desc)

        mocked_delete_oldest_points.assert_called_once()
        mocked_delete_empty_tracks_rows.assert_called_once()
        mocked_nn_match_two_way.assert_not_called()
        mocked_append_tracks.assert_called_once()
        self.assertIsNone(mocked_append_tracks.call_args[1]['matched'])
        np.testing.assert_array_equal(mocked_append_tracks.call_args[1]['unmatched'], np.arange(pts.shape[1]))
        np.testing.assert_array_equal(self.tracker.prev_desc, desc)

    @patch('models.model_wrap.PointTracker2.append_tracks')
    @patch('models.model_wrap.PointTracker2.nn_match_two_way')
    @patch('models.model_wrap.PointTracker2.delete_empty_tracks_rows')
    @patch('models.model_wrap.PointTracker2.delete_oldest_points')
    def test_7_2(
            self,
            mocked_delete_oldest_points,
            mocked_delete_empty_tracks_rows,
            mocked_nn_match_two_way,
            mocked_append_tracks
    ):
        """ Tests update: case non-first call """
        pts1 = np.array([
            [1.0, 1.0, 2.00, 2.00, 3.00],  # x_i
            [1.0, 2.0, 2.00, 1.00, 2.00],  # y_i
            [0.7, 0.8, 0.74, 0.75, 0.76],  # confidence_i
        ])
        desc1 = np.full((10, 5), 2)
        pts2 = np.array([
            [11.0, 11.0, 12.00, 12.00, 13.00],  # x_i
            [11.0, 12.0, 12.00, 11.00, 22.00],  # y_i
            [0.10, 0.20, 0.340, 0.350, 0.360],  # confidence_i
        ])
        desc2 = np.full((10, 5), 3)

        matches_found = np.array([
            [0., 3.],
            [0., 4.],
            [.1, .5],
        ])

        # first call
        self.tracker.update(pts1, desc1)
        mocked_delete_oldest_points.reset_mock()
        mocked_delete_empty_tracks_rows.reset_mock()
        mocked_nn_match_two_way.reset_mock()
        mocked_nn_match_two_way.return_value = matches_found
        mocked_append_tracks.reset_mock()

        # second call
        self.tracker.update(pts2, desc2)
        mocked_delete_oldest_points.assert_called_once()
        mocked_delete_empty_tracks_rows.assert_called_once()
        mocked_nn_match_two_way.assert_called_once()
        np.testing.assert_array_equal(mocked_nn_match_two_way.call_args[0][0], desc1)
        np.testing.assert_array_equal(mocked_nn_match_two_way.call_args[0][1], desc2)
        self.assertEqual(mocked_nn_match_two_way.call_args[0][2], self.desc_threshold)
        mocked_append_tracks.assert_called_once()
        np.testing.assert_array_equal(mocked_append_tracks.call_args[1]['matched'], matches_found)
        np.testing.assert_array_equal(mocked_append_tracks.call_args[1]['unmatched'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(self.tracker.prev_desc, desc2)

    def test_7_3(self):
        """ Tests update: integration test """
        self.num_frames = 3
        self.desc_threshold = .2
        self.tracker = PointTracker2(self.num_frames, self.desc_threshold)

        pts1 = np.array([
            [1.0, 1.0, 2.00, 2.00, 3.00],  # x_i
            [1.0, 2.0, 2.00, 1.00, 2.00],  # y_i
            [0.7, 0.8, 0.74, 0.75, 0.76],  # confidence_i
        ])
        desc1 = np.array([[0.61, 0.37, 0.28, 0.09, .5],
                          [0.44, 0.04, 0.96, 0.29, .1],
                          [0.27, 0.89, 0.80, 0.34, .7],
                          [0.26, 0.50, 0.33, 0.90, .3],
                          [0.93, 0.63, 0.52, 0.17, .4]])
        desc1_unit_norm = desc1 / np.linalg.norm(desc1, axis=0)

        # first call - only unmathed descriptors ##############################
        expected_tracks = np.array([
            [0, 0, 0, -1, -1, 0],
            [1, 0, 0, -1, -1, 1],
            [2, 0, 0, -1, -1, 2],
            [3, 0, 0, -1, -1, 3],
            [4, 0, 0, -1, -1, 4],
        ])

        self.tracker.update(pts1, desc1_unit_norm)
        np.testing.assert_array_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_equal(self.tracker.prev_desc, desc1_unit_norm)

        # second call - matched and unmatched #################################
        pts2 = np.array([
            [11.0, 11.0, 12.00, 12.00],  # x_i
            [11.0, 12.0, 12.00, 11.00],  # y_i
            [0.10, 0.20, 0.340, 0.350],  # confidence_i
        ])
        desc2 = desc1[:, [0, 2, 3, 4]].copy()
        desc2[:, 1] += .3
        desc2[:, 2] += .9
        desc2[:, 3] += .2
        desc2_unit_norm = desc2 / np.linalg.norm(desc2, axis=0)

        expected_matched_desc1_idxs = [0, 2, 4]
        expected_matched_desc2_idxs = [0, 1, 3]
        expected_unmatched_desc2_idxs = [2]
        dmat = cdist(desc1_unit_norm.T, desc2_unit_norm.T, metric='euclidean')
        ca = 1  # tracks col 1
        n = 2  # tracks col 2
        # 0  is in tracks[0][-1]
        ca2_0_0 = (dmat[0, 0] + (expected_tracks[0, n]+1)*expected_tracks[0, ca])/(expected_tracks[0, n]+2)
        # 2  is in tracks[2][-1]
        ca2_2_1 = (dmat[2, 1] + (expected_tracks[2, n]+1)*expected_tracks[2, ca])/(expected_tracks[2, n]+2)
        # 4  is in tracks[4][-1]
        ca2_4_3 = (dmat[4, 3] + (expected_tracks[4, n]+1)*expected_tracks[4, ca])/(expected_tracks[4, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [0, ca2_0_0, 1, -1,  0,  0],
            [1,       0, 0, -1,  1, -1],
            [2, ca2_2_1, 1, -1,  2,  1],
            [3,       0, 0, -1,  3, -1],
            [4, ca2_4_3, 1, -1,  4,  3],
            # new tracked descriptors
            [5,       0,  0, -1, -1, 2]
        ])

        self.tracker.update(pts2, desc2_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc2_unit_norm)

        # third call - only mached descriptors ################################
        pts3 = np.array([
            [5.0,  22],  # x_i
            [5.0,  12],  # y_i
            [.33, .42],  # confidence_i
        ])
        desc3 = desc2[:, [2, 1]].copy()
        desc3[:, 0] -= .15  # desc idx 2
        desc3[:, 1] -= .3   # desc idx 1
        desc3_unit_norm = desc3 / np.linalg.norm(desc3, axis=0)

        expected_matched_desc2_idxs = [2, 1]
        expected_matched_desc3_idxs = [0, 1]
        expected_unmatched_desc3_idxs = []

        dmat = cdist(desc2_unit_norm.T, desc3_unit_norm.T, metric='euclidean')
        # 2  is in tracks[5][-1]
        ca3_2_0 = (dmat[2, 0] + (expected_tracks[5, n]+1)*expected_tracks[5, ca])/(expected_tracks[5, n]+2)
        # 1 is in tracks[2][-1]
        ca3_1_1 = (dmat[1, 1] + (expected_tracks[2, n]+1)*expected_tracks[2, ca])/(expected_tracks[2, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [0, ca2_0_0, 1, 0,  0, -1],
            [1,       0, 0, 1, -1, -1],
            [2, ca3_1_1, 2, 2,  1,  1],
            [3,       0, 0, 3, -1, -1],
            [4, ca2_4_3, 1, 4,  3, -1],
            [5, ca3_2_0, 1, -1, 2,  0],
            # new tracked descriptors
        ])

        self.tracker.update(pts3, desc3_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc3_unit_norm)

        # fourth call - some tracks/rows deletions ############################
        pts4 = np.array([
            [6.0],  # x_i
            [6.0],  # y_i
            [.93],  # confidence_i
        ])
        desc4 = np.array([[.77],
                          [.66],
                          [.08],
                          [.21],
                          [.61]])
        desc4_unit_norm = desc4 / np.linalg.norm(desc4, axis=0)
        expected_tracks = np.array([
            # matched descriptors
            [0, ca2_0_0, 1,  0, -1, -1],
            [2, ca3_1_1, 2,  1,  1, -1],
            [4, ca2_4_3, 1,  3, -1, -1],
            [5, ca3_2_0, 1,  2,  0, -1],
            # new tracked descriptors
            [6,       0, 0, -1, -1,  0],
        ])

        self.tracker.update(pts4, desc4_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc4_unit_norm)

        # fifth call - only unmatched #########################################
        pts5 = np.array([
            [26.0, 98],  # x_i
            [6.0, 77],  # y_i
            [.12,  .5],  # confidence_i
        ])
        desc5 = np.array([[.09, .99],
                          [.07, .98],
                          [.06, .97],
                          [.05, .96],
                          [.04, .95],])
        desc5_unit_norm = desc5 / np.linalg.norm(desc5, axis=0)
        expected_tracks = np.array([
            # matched descriptors
            [2, ca3_1_1, 2,   1, -1, -1],
            [5, ca3_2_0, 1,   0, -1, -1],
            [6,       0, 0,  -1,  0, -1],
            # new tracked descriptors
            [7,       0, 0,  -1, -1,  0],
            [8,       0, 0,  -1, -1,  1],
        ])

        self.tracker.update(pts5, desc5_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc5_unit_norm)

        # sixth call - only matched ###########################################
        pts6 = np.array([
            [101],  # x_i
            [78],  # y_i
            [.67],  # confidence_i
        ])
        desc6 = desc5[:, 1].copy()
        desc6 = desc6[:, np.newaxis]
        desc6[:, 0] += .22
        desc6_unit_norm = desc6 / np.linalg.norm(desc6, axis=0)

        expected_matched_desc5_idxs = [1]
        expected_matched_desc6_idxs = [0]
        expected_unmatched_desc6_idxs = []

        dmat = cdist(desc5_unit_norm.T, desc6_unit_norm.T, metric='euclidean')
        # 1  is in tracks[4][-1]
        ca6_1_0 = (dmat[1, 0] + (expected_tracks[4, n]+1)*expected_tracks[4, ca])/(expected_tracks[4, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [6,       0, 0,  0, -1, -1],
            [7,       0, 0, -1,  0, -1],
            [8, ca6_1_0, 1, -1,  1,  0],
            # new tracked descriptors
        ])

        self.tracker.update(pts6, desc6_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc6_unit_norm)

        # seventh call - matched and unmatched ################################
        pts7 = np.array([
            [1,  2,   3],  # x_i
            [7, 20,  21],  # y_i
            [.7, .3, .73],  # confidence_i
        ])
        desc7 = np.array([[.37, .01],
                          [.23, .02],
                          [.36, .03],
                          [.11, .44],
                          [.09, .45]])
        desc7 = np.hstack([desc7.copy(), desc6.copy()])
        desc7[:, 2] -= .31
        desc7_unit_norm = desc7 / np.linalg.norm(desc7, axis=0)

        expected_matched_desc6_idxs = [0]
        expected_matched_desc7_idxs = [2]
        expected_unmatched_desc7_idxs = [0, 1]

        dmat = cdist(desc6_unit_norm.T, desc7_unit_norm.T, metric='euclidean')
        # 0  is in tracks[2][-1]
        ca7_0_2 = (dmat[0, 2] + (expected_tracks[2, n]+1)*expected_tracks[2, ca])/(expected_tracks[2, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [7,        0, 0,  0, -1, -1],
            [8,  ca7_0_2, 2,  1,  0,  2],
            # new tracked descriptors
            [9,        0, 0, -1, -1, 0],
            [10,       0, 0, -1, -1, 1],
        ])

        self.tracker.update(pts7, desc7_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc7_unit_norm)

        # eight call - unmatched ##############################################
        pts8 = np.array([
            [51,  29,  13],  # x_i
            [41,  17,  12],  # y_i
            [.1, .35, .37],  # confidence_i
        ])
        desc8 = np.array([[.55, .21, .87],
                          [.12, .12, .17],
                          [.17, .54, .92],
                          [.01, .45, .77],
                          [.99, .44, .01]])
        desc8_unit_norm = desc8 / np.linalg.norm(desc8, axis=0)

        expected_tracks = np.array([
            # matched descriptors
            [8,  ca7_0_2, 2,   0, 2, -1],
            [9,        0, 0,  -1, 0, -1],
            [10,       0, 0,  -1, 1, -1],
            # new tracked descriptors
            [11,       0, 0,  -1, -1, 0],
            [12,       0, 0,  -1, -1, 1],
            [13,       0, 0,  -1, -1, 2],
        ])

        self.tracker.update(pts8, desc8_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc8_unit_norm)

        # nineth call - only matched ##########################################
        pts9 = np.array([
            [151,  129,  113],  # x_i
            [141,  117,  112],  # y_i
            [.11, .135, .137],  # confidence_i
        ])
        desc9 = desc8[:, [2, 0, 1]].copy()
        desc9[:, 0] += .13
        desc9[:, 1] += .15  # .3 TODO: find out why the distance increases so much with .3
        desc9[:, 2] += .33
        desc9_unit_norm = desc9 / np.linalg.norm(desc9, axis=0)

        expected_matched_desc8_idxs = [2, 0, 1]
        expected_matched_desc9_idxs = [0, 1, 2]
        expected_unmatched_desc9_idxs = []

        dmat = cdist(desc8_unit_norm.T, desc9_unit_norm.T, metric='euclidean')

        # 2  is in tracks[5][-1]
        ca9_2_0 = (dmat[2, 0] + (expected_tracks[5, n]+1)*expected_tracks[5, ca])/(expected_tracks[5, n]+2)
        # 0  is in tracks[3][-1]
        ca9_0_1 = (dmat[0, 1] + (expected_tracks[3, n]+1)*expected_tracks[3, ca])/(expected_tracks[3, n]+2)
        # 1  is in tracks[4][-1]
        ca9_1_2 = (dmat[1, 2] + (expected_tracks[4, n]+1)*expected_tracks[4, ca])/(expected_tracks[4, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [8,  ca7_0_2, 2,  2, -1, -1],
            [9,        0, 0,  0, -1, -1],
            [10,       0, 0,  1, -1, -1],
            [11, ca9_0_1, 1, -1,  0,  1],
            [12, ca9_1_2, 1, -1,  1,  2],
            [13, ca9_2_0, 1, -1,  2,  0],
            # new tracked descriptors
        ])

        self.tracker.update(pts9, desc9_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc9_unit_norm)

        # tenth call - matched and unmatched ##################################
        pts10 = np.array([
            [11,  19,  13],  # x_i
            [11,  17,  12],  # y_i
            [.1, .15, .17],  # confidence_i
        ])
        desc10 = np.array([[.98, .222],
                           [.89, .221],
                           [.77, .232],
                           [.66, .231],
                           [.11, .233]])
        desc10 = np.hstack([desc10, desc9[:, 1][:, np.newaxis]])
        desc10[:, 2] += .1
        desc10_unit_norm = desc10 / np.linalg.norm(desc10, axis=0)

        expected_matched_desc9_idxs = [1]
        expected_matched_desc10_idxs = [2]
        expected_unmatched_desc10_idxs = [0, 1]

        dmat = cdist(desc9_unit_norm.T, desc10_unit_norm.T, metric='euclidean')

        # 1 is in tracks[3][-1]
        ca10_1_2 = (dmat[1, 2] + (expected_tracks[3, n]+1)*expected_tracks[3, ca])/(expected_tracks[3, n]+2)

        expected_tracks = np.array([
            # matched descriptors
            [11, ca10_1_2, 2,  0,  1,  2],
            [12,  ca9_1_2, 1,  1,  2, -1],
            [13,  ca9_2_0, 1,  2,  0, -1],
            # new tracked descriptors
            [14,        0, 0, -1, -1,  0],
            [15,        0, 0, -1, -1,  1],
        ])

        self.tracker.update(pts10, desc10_unit_norm)
        np.testing.assert_array_almost_equal(self.tracker.tracks, expected_tracks)
        np.testing.assert_array_almost_equal(self.tracker.prev_desc, desc10_unit_norm)

    @patch('models.model_wrap.PointTracker2.update')
    def test_8(self, mocked_update):
        """ Tests __call__ """
        pts = np.array([
            [1.0, 1.0, 2.00, 2.00, 3.00],  # x_i
            [1.0, 2.0, 2.00, 1.00, 2.00],  # y_i
            [0.7, 0.8, 0.74, 0.75, 0.76],  # confidence_i
        ])
        desc = np.empty((10, 5))
        self.tracker.update(pts, desc)

        mocked_update.assert_called_once()
        np.testing.assert_array_equal(mocked_update.call_args[0][0], pts)
        np.testing.assert_array_equal(mocked_update.call_args[0][1], desc)

    def test_9_1(self):
        """ Tests nn_match_two_way: case desc1.shape[1] == 0 """
        desc1 = np.empty((5, 0))
        desc2 = np.random.rand(5, 4)
        desc1 /= np.linalg.norm(desc1, axis=0)
        desc2 /= np.linalg.norm(desc2, axis=0)
        nn_thresh = .2
        matches = self.tracker.nn_match_two_way(desc1, desc2, nn_thresh)
        expected_matches = np.zeros((3, 0))
        np.testing.assert_array_equal(matches, expected_matches)

    def test_9_2(self):
        """ Tests nn_match_two_way: case desc2.shape[1] == 0 """
        desc1 = np.random.rand(5, 5)
        desc2 = np.empty((5, 0))
        desc1 /= np.linalg.norm(desc1, axis=0)
        desc2 /= np.linalg.norm(desc2, axis=0)
        nn_thresh = .2
        matches = self.tracker.nn_match_two_way(desc1, desc2, nn_thresh)
        expected_matches = np.zeros((3, 0))
        np.testing.assert_array_equal(matches, expected_matches)

    def test_9_3(self):
        """ Tests nn_match_two_way: standard case """
        desc1 = np.array([[0.61598891, 0.37766321, 0.28693948, 0.09227691, .5],
                          [0.44496694, 0.04681025, 0.96239828, 0.29838603, .1],
                          [0.27661612, 0.8987756, 0.80782054, 0.34033582, .7],
                          [0.26759243, 0.50839173, 0.33474996, 0.90825325, .3],
                          [0.93652361, 0.63927898, 0.52164754, 0.17623799, .4]])
        desc2 = desc1[:, [0, 2, 3, 4]].copy()
        desc2[:, 1] += .3
        desc2[:, 2] += .9
        desc2[:, 3] += .2
        desc1 /= np.linalg.norm(desc1, axis=0)
        desc2 /= np.linalg.norm(desc2, axis=0)
        nn_thresh = .2
        matches = self.tracker.nn_match_two_way(desc1, desc2, nn_thresh)
        expected_desc1_idx = [0, 2, 4]
        expected_desc2_idx = [0, 1, 3]
        expected_scores = [
            0,
            cdist(desc1[:, 2][:, np.newaxis].T, desc2[:, 1][:, np.newaxis].T, metric='euclidean')[0][0],
            cdist(desc1[:, 4][:, np.newaxis].T, desc2[:, 3][:, np.newaxis].T, metric='euclidean')[0][0]
        ]
        expected_matches = np.vstack([expected_desc1_idx, expected_desc2_idx, expected_scores])

        np.testing.assert_array_almost_equal(matches, expected_matches)

    def test_10_1(self):
        """ Tests to_hashmap_by_value: standard case """
        array = np.array([1, 4, 3, 0, 2, 5])
        expected_hashmap = {
            1: 0,
            4: 1,
            3: 2,
            0: 3,
            2: 4,
            5: 5,
        }
        actual_hashmap = self.tracker.to_hashmap_by_value(array)

        self.assertEqual(actual_hashmap, expected_hashmap)

    def test_10_2(self):
        """ Tests to_hashmap_by_value: case repeated values """
        array = np.array([1, 4, 3, 0, 3, 1, 4, 9])
        expected_hashmap = {
            1: 5,
            4: 6,
            3: 4,
            0: 3,
            9: 7
        }
        actual_hashmap = self.tracker.to_hashmap_by_value(array)

        self.assertEqual(actual_hashmap, expected_hashmap)

    def test_10_3(self):
        """ Tests to_hashmap_by_value: empty input """
        array = np.empty(0)
        expected_hashmap = {}
        actual_hashmap = self.tracker.to_hashmap_by_value(array)

        self.assertEqual(actual_hashmap, expected_hashmap)

    def test_11_1(self):
        """ Tests get_new_column_for_appending: case full match """
        matches = np.array([
            [13, 10, 12, 14, 11],  # idxs descriptor 1
            [3,   0,  2,  4,  1],  # idxs descriptor 2
        ])

        expected_new_col = [0, 1, 2, 3, 4]
        expected_idxs_updated = [3, 0, 2, 4, 1]
        actual_new_col, actual_idxs_updated = self.tracker.get_new_column_for_appending(matches)

        np.testing.assert_array_equal(actual_new_col, expected_new_col)
        self.assertEqual(actual_idxs_updated, expected_idxs_updated)

    def test_11_2(self):
        """ Tests get_new_column_for_appending: case non-full match """
        matches = np.array([
            [14, 11, 13],  # idxs descriptor 1
            [4,  1,  3],  # idxs descriptor 2
        ])

        expected_new_col = [-1, 1, -1, 3, 4]
        expected_idxs_updated = [4, 1, 3]
        actual_new_col, actual_idxs_updated = self.tracker.get_new_column_for_appending(matches)

        np.testing.assert_array_equal(actual_new_col, expected_new_col)
        self.assertEqual(actual_idxs_updated, expected_idxs_updated)

    def test_11_3(self):
        """ Tests get_new_column_for_appending: case no matches """
        matches = np.array([
            [-1, -1, -1, -1, -1],  # idxs descriptor 1
            [-1, -1, -1, -1, -1],  # idxs descriptor 2
        ])

        expected_new_col = [-1, -1, -1, -1, -1]
        expected_idxs_updated = []
        actual_new_col, actual_idxs_updated = self.tracker.get_new_column_for_appending(matches)

        np.testing.assert_array_equal(actual_new_col, expected_new_col)
        self.assertEqual(actual_idxs_updated, expected_idxs_updated)

    def test_11_4(self):
        """ Tests get_new_column_for_appending: case empty matches matrix """
        matches = np.empty((2, 0))
        expected_new_col = [-1, -1, -1, -1, -1]
        expected_idxs_updated = []

        actual_new_col, actual_idxs_updated = self.tracker.get_new_column_for_appending(matches)

        np.testing.assert_array_equal(actual_new_col, expected_new_col)
        self.assertEqual(actual_idxs_updated, expected_idxs_updated)

    def test_12(self):
        """ clear_desc """
        self.tracker.prev_desc = np.random.rand(5, 3)
        self.tracker.clear_desc()
        self.assertIsNone(self.tracker.prev_desc)


if __name__ == "__main__":
    unittest.main()
