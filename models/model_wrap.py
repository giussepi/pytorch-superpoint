# -*- coding: utf-8 -*-
""" models/model_wrap.py

class to process superpoint net
# may be some duplication with model_wrap.py
# PointTracker is from Daniel's repo.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn, Tensor

from models.SuperPointNet_pretrained import SuperPointNet
from utils.loader import modelLoader
from utils.losses import extract_patch_from_points, soft_argmax_2d, norm_patches, do_log
from utils.utils import toNumpy, extract_points, flattenDetection

__all__ = [
    'labels2Dto3D',
    'SuperPointFrontend_torch',
    'PointTracker',
]


def labels2Dto3D(cell_size, labels):
    H, W = labels.shape[0], labels.shape[1]
    Hc, Wc = H // cell_size, W // cell_size
    labels = labels[:, np.newaxis, :, np.newaxis]
    labels = labels.reshape(Hc, cell_size, Wc, cell_size)
    labels = np.transpose(labels, [1, 3, 0, 2])
    labels = labels.reshape(1, cell_size ** 2, Hc, Wc)
    labels = labels.squeeze()
    dustbin = labels.sum(axis=0)
    dustbin = 1 - dustbin
    dustbin[dustbin < 0] = 0
    labels = np.concatenate((labels, dustbin[np.newaxis, :, :]), axis=0)

    return labels


class SuperPointFrontend_torch:
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, config, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False, trained=False, device='cpu', grad=False, load=True):
        self.config = config

        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.patches = None

        self.device = device
        self.subpixel = False
        if self.config['model']['subpixel']['enable']:
            self.subpixel = True

        if load:
            self.loadModel(weights_path)

    def loadModel(self, weights_path):
        # Load the network in inference mode.
        if weights_path[-4:] == '.tar':
            model = self.config['model']['name']
            params = self.config['model']['params']
            print("model: ", model)
            self.net = modelLoader(model=model, **params)
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net = SuperPointNet()
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

        self.net = self.net.to(self.device)

    def net_parallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)

    @property
    def points(self):
        print("get pts")
        return self.pts

    @property
    def heatmap(self):
        # print("get heatmap")
        return self._heatmap

    @heatmap.setter
    def heatmap(self, heatmap):
        # print("set heatmap")
        self._heatmap = heatmap

    def soft_argmax_points(self, pts, patch_size: int = 5, verbose: bool = False):
        """
        input:
            pts: tensor [N x 2]
        """
        ##### check not take care of batch #####
        # print("not take care of batch! only take first element!")
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(self.heatmap, pts, patch_size=patch_size, verbose=verbose)
        patches = np.stack(patches)
        patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)

        # norm patches
        patches_torch = norm_patches(patches_torch)
        patches_torch = do_log(patches_torch)

        dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
        points = pts
        points[:, :2] = points[:, :2] + dxdy.numpy().squeeze() - patch_size//2
        self.patches = patches_torch.numpy().squeeze()
        self.pts_subpixel = [points.transpose().copy()]

        return self.pts_subpixel.copy()

    @staticmethod
    def get_image_patches(pts, image, patch_size=5):
        """
        input:
            image: np [H, W]
        return:
            patches: np [N, patch, patch]
        """
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(image, pts, patch_size=patch_size)
        patches = np.stack(patches)

        return patches

    def extract_points(self, heatmap: np.ndarray | Tensor):
        return extract_points(heatmap, self.conf_thresh, self.nms_dist, self.border_remove)

    def sample_desc_from_points(self, coarse_desc, pts):
        # --- Process descriptor.
        H, W = coarse_desc.shape[2]*self.cell, coarse_desc.shape[3]*self.cell
        D = coarse_desc.shape[1]

        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        return desc

    def subpixel_predict(self, pred_res, points, verbose=False):
        """
        input:
            labels_res: numpy [2, H, W]
            points: [3, N]
        return:
            subpixels: [3, N]
        """
        D = points.shape[0]

        if points.shape[1] == 0:
            pts_subpixel = np.zeros((D, 0))
        else:
            points_res = pred_res[:, points[1, :].astype(int), points[0, :].astype(int)]
            pts_subpixel = points.copy()
            if verbose:
                print("before: ", pts_subpixel[:, :5])
            pts_subpixel[:2, :] += points_res
            if verbose:
                print("after: ", pts_subpixel[:, :5])

        return pts_subpixel

    def run(self, image: np.ndarray, only_heatmap: bool = False, train: bool = True) -> tuple:
        """
        Extracts points and descriptors from the input image

        Kwargs:
            image  <np.ndarray>: HxW tensor float32 input image in range [0,1].
            only_heatmap <bool>: Whether or not return the heatmap only.
                                 Default False
            train        <bool>:

        Returns:
            pts <list[np.ndarray]>: list containing a batch of 3xN numpy array corners
                                    [x_i, y_i, confidence_i]
            pts_desc <list[np.ndarray]>: list containing a batch of 256xN numpy array descriptors
            dense_desc <Tensor>: Tensor [B, 256, H, W] of corresponding unit normalized descriptors.
            heatmap <Tensor>: Tensor [B, 1, H, W] of confidence scores in [0,1].
        """
        inp = image.to(self.device)
        batch_size, H, W = inp.shape[0], inp.shape[2], inp.shape[3]

        if train:
            outs = self.net.forward(inp)
            semi, coarse_desc = outs['semi'], outs['desc']
        else:
            with torch.no_grad():
                outs = self.net.forward(inp)
                semi, coarse_desc = outs['semi'], outs['desc']

        # as tensor
        # flatten detection
        heatmap = flattenDetection(semi, tensor=True)
        self.heatmap = heatmap

        if only_heatmap:
            return heatmap

        # extract keypoints
        pts = [self.extract_points(heatmap[i, :, :, :].cpu().detach().numpy().squeeze()) for i in range(batch_size)]
        self.pts = pts

        if self.subpixel:
            # FIXME: the following line is not working
            labels_res = outs[2]
            self.pts_subpixel = [self.subpixel_predict(toNumpy(labels_res[i, ...]), pts[i]) for i in range(batch_size)]

        # interpolate description
        '''
        coarse_desc:
            tensor (Batch_size, 256, Hc, Wc)
        dense_desc:
            tensor (batch_size, 256, H, W)
        '''
        dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(self.cell, self.cell), mode='bilinear')
        # norm the descriptor

        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc
        dense_desc = norm_desc(dense_desc)

        # extract descriptors
        dense_desc_cpu = dense_desc.cpu().detach().numpy()
        pts_desc = [dense_desc_cpu[i, :, pts[i][1, :].astype(
            int), pts[i][0, :].astype(int)].transpose() for i in range(len(pts))]

        if self.subpixel:
            return self.pts_subpixel, pts_desc, dense_desc, heatmap

        return pts, pts_desc, dense_desc, heatmap


class PointTracker:
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length=2, nn_thresh=0.7):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for _ in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        self.matches = None
        # self.last_pts = None
        self.mscores = None

    def nn_match_two_way(self, desc1: np.ndarray, desc2: np.ndarray, nn_thresh: float) -> np.ndarray:
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Kwargs:
            desc1 <np.ndarray>: MxN matrix of N M-dimensional descriptors.
            desc2 <np.ndarray>: MxP matrix of P M-dimensional descriptors.
            nn_thresh  <float>: Maximum descritor distance

        Returns:
            matches <np.ndarray>: 3xL matrix, of L matches, where L <= N and each column i is
                                  a match of two descriptors, idx_i index from desc1 and
                                  idx_j index from desc2: [idx_i, idx_j, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        assert nn_thresh > 0, nn_thresh

        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))

        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)  # NxO
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)  # N positions selected from desc2
        scores = dmat[np.arange(dmat.shape[0]), idx]  # N
        # Threshold the NN matches.
        keep = scores < nn_thresh  # N
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)  # O positions selected from desc1
        keep_bi = np.arange(len(idx)) == idx2[idx]  # FIXME: case min distance links to more than one point
        keep = np.logical_and(keep, keep_bi)  # N
        # FIXME: the previous line could be replaced by.
        # keep = keep * keep_bi
        idx = idx[keep]  # <= N
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        self.mscores = matches

        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)

        return offsets

    def get_matches(self):
        return self.matches

    def get_mscores(self):
        return self.mscores

    def clear_desc(self):
        self.last_desc = None

    def update(self, pts: np.ndarray, desc: np.ndarray):
        """ Add a new set of point and descriptor observations to the tracker.

        Kwargs:
            pts  <np.ndarray>: 3xN matrix of 2D point observations with format [x_i, y_i, confidence_i].
            desc <np.ndarray>: DxN matrix of D dimensional descriptors corresponding to pts.
        """
        assert isinstance(pts, np.ndarray), type(pts)
        assert isinstance(desc, np.ndarray), type(desc)
        assert pts.size > 0
        assert desc.size > 0
        assert pts.shape[1] == desc.shape[1], f'pts.shape={pts.shape}, desc.shape={desc.shape}'

        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))

        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        self.matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        # pts_id = pts[:2, :]  # [x_i, y_i]

        # NOTE: this is wrong because the subsequent for loop employs offsets to update the indexes,
        #       and the following lines replaces the 3xL indexes matrix with a 4xL points matrix.
        #       As a result, keeping the following lines will make the algorithm to wrongly update points
        #       with index offsets.
        # if self.last_pts is not None:
        #     id1 = self.last_pts[:, self.matches[0, :].astype(int)]  # matched previous pts x_i,y_i [2,R]
        #     id2 = pts_id[:, self.matches[1, :].astype(int)]  # matched new pts x_i,y_i  [2,R]
        #     self.matches = np.concatenate((id1, id2), axis=0)  # [4,R]
        # FIXME: Commenting the above lines makes get_matches and get_scores return the same matrix.
        #        This change requires modification in the evaluation.py script

        for match_ in self.matches.T:
            # Add a new point to it's matched track.
            id1 = match_[0] + offsets[-2]  # updated old descriptor idx
            id2 = match_[1] + offsets[-1]  # updated new descriptor idx
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match_[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2  # adding new descriptor idx as the latest point in the row
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match_[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match_[2]

        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        # self.last_pts = pts[:2, :].copy()  # [x_i, y_i]


class PointTracker2:
    """
    Contains methods to track points based on the L2 distance between their descriptors.

    Usage:
        tracker = PointTracker()
        tracker.update(<3xN array of N points>, <DxN array of N points' descriptors>)
        # or
        tracker(<3xN array of N points>, <DxN array of N points' descriptors>)
    """

    def __init__(
            self, num_frames: int = 2, desc_threshold: float = .7, /, *,
            max_points: int = -1, min_conf: float = .5
    ):
        """
        Kwargs:
            num_frames       <int>: number of frames to track.
                                    Default 2
            desc_threshold <float>: maximum descriptors L2 distance allowed
                                    Default .7
            TODO: implement max_points logic
            max_points      <int> : maximum number of points to track (>=1). Set it to -1 for unlimited tracks
                                    Default -1.
            # TODO: implement it
            min_conf       <float>: minimum confidence of points
                                    Default .5
        """
        assert isinstance(num_frames, int), type(num_frames)
        assert num_frames >= 2, num_frames
        assert isinstance(desc_threshold, float), type(desc_threshold)
        assert isinstance(max_points, int), type(max_points)
        max_points = max_points if max_points >= 0 else np.inf
        assert max_points > 0, max_points
        assert 0 < min_conf < 1, min_conf

        self.num_frames = num_frames
        self.desc_threshold = desc_threshold
        self.prev_desc = None
        self.tracks = np.empty((0, self.num_frames + 3))
        # NOTE: tracks columns are:
        # |----------+----------------+----------+------------+-----+---------------------|
        # | track_ID | cumulative_avg | num_desc | desc_ids_1 | ... | desc_ids_num_frames |
        # |----------+----------------+----------+------------+-----+---------------------|
        # Where
        # track_ID       : tracking ID
        # cumulative_avg : cumulative average of scores (descriptors L2 distance)
        # num_desc       : n value utilised to compute the cumulative_avg, i.e., zero-based counter
        #                  of points/descriptors processed (so it means processed points - 1)
        # desc_ids_n     : descriptors ids/positions tracked at time n, where  2 <= n <= num_frames
        self.total_tracks = 0  # all-time tracked descriptors/points
        self.max_points = max_points
        self.min_conf = min_conf

    def __call__(self, pts: np.ndarray, desc: np.ndarray):
        """
        Calls upddate to add points and descriptors to tracks

        Kwargs:
            pts  <np.ndarray>: 3xN array of N points with shape [x_n, y_n, confidence_n]
            desc <np.ndarray>: DxN array of corresponding N D-dimensional descriptors
        """
        return self.update(pts, desc)

    # @staticmethod
    # def nn_match_two_way(desc1: np.ndarray, desc2: np.ndarray, nn_thresh: float) -> np.ndarray:
    #     # TODO: need to be improved
    #     """
    #     Performs two-way nearest neighbor matching of two sets of descriptors, such
    #     that the NN match from descriptor A->B must equal the NN match from B->A.

    #     Kwargs:
    #         desc1 <np.ndarray>: MxN matrix of N M-dimensional descriptors.
    #         desc2 <np.ndarray>: MxP matrix of P M-dimensional descriptors.
    #         nn_thresh  <float>: Maximum descritor distance

    #     Returns:
    #         matches <np.ndarray>: 3xL matrix, of L matches, where L <= N and each column i is
    #                               a match of two descriptors, idx_i index from desc1 and
    #                               idx_j index from desc2: [idx_i, idx_j, match_score]^T
    #     """
    #     assert desc1.shape[0] == desc2.shape[0]
    #     assert nn_thresh > 0, nn_thresh

    #     if desc1.shape[1] == 0 or desc2.shape[1] == 0:
    #         return np.zeros((3, 0))

    #     # Compute L2 distance. Easy since vectors are unit normalized.
    #     dmat = np.dot(desc1.T, desc2)  # NxP
    #     dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    #     # from scipy.spatial.distance import cdist
    #     # cdist(desc1.T, desc1.T, metric='euclidean')
    #     # or
    #     # cosine distance
    #     # dmat = 1 - np.matmul(desc1.T, desc2)
    #     # Get NN indices and scores.
    #     idx = np.argmin(dmat, axis=1)  # N positions selected from desc2
    #     scores = dmat[np.arange(dmat.shape[0]), idx]  # N
    #     # Threshold the NN matches.
    #     keep = scores < nn_thresh  # N
    #     # Check if nearest neighbor goes both directions and keep those.
    #     idx2 = np.argmin(dmat, axis=0)  # P positions selected from desc1
    #     keep_bi = np.arange(len(idx)) == idx2[idx]  # FIXME: case min distance links to more than one point
    #     keep = np.logical_and(keep, keep_bi)  # N
    #     # FIXME: the previous line could be replaced by.
    #     # keep = keep * keep_bi
    #     idx = idx[keep]  # <= N
    #     scores = scores[keep]
    #     # Get the surviving point indices.
    #     m_idx1 = np.arange(desc1.shape[1])[keep]
    #     m_idx2 = idx
    #     # Populate the final 3xL match data structure. L <= N
    #     matches = np.zeros((3, int(keep.sum())))
    #     matches[0, :] = m_idx1
    #     matches[1, :] = m_idx2
    #     matches[2, :] = scores

    #     return matches

    @staticmethod
    def nn_match_two_way(
            desc1: np.ndarray, desc2: np.ndarray, nn_thresh: float, /, *,  cosine_distance: bool = False
    ) -> np.ndarray:
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Kwargs:
            desc1     <np.ndarray>: MxN matrix of N M-dimensional unit normalized descriptors.
            desc2     <np.ndarray>: MxP matrix of P M-dimensional unit normalized descriptors.
            nn_thresh      <float>: Maximum distance among descriptors.
            cosine_distance <bool>: Employs cosine distance if true, else, L2 distance is utilized
                                    Default False

        Returns:
            matches <np.ndarray>: 3xS matrix, of S matches, where S <= N and each column i is
                                  a match of two descriptors, idx_0_s index from desc1 and
                                  idx_1_s index from desc2: [idx_0_s, idx_1_s, match_score_2_s]^T
        """
        assert isinstance(desc1, np.ndarray), type(desc1)
        assert isinstance(desc2, np.ndarray), type(desc2)
        assert isinstance(nn_thresh, float), type(nn_thresh)
        assert isinstance(cosine_distance, bool), type(cosine_distance)
        assert desc1.shape[0] == desc2.shape[0]
        assert nn_thresh > 0, nn_thresh

        N, P = desc1.shape[1], desc2.shape[1]

        if N == 0 or P == 0:
            return np.zeros((3, 0))

        # computes distances NxP matrix
        if cosine_distance:
            dmat = 1 - np.matmul(desc1.T, desc2)
        else:  # Euclidean distance from unit normalized vectors
            dmat = cdist(desc1.T, desc2.T, metric='euclidean')  # NxP distance matrix

        idx_desc2 = np.argmin(dmat, axis=1)  # N positions selected from desc2
        scores = dmat[np.arange(N), idx_desc2]  # N distance scores
        selection = scores <= nn_thresh  # Thresholded selection. N-dimensional array

        # Applying bidirectional nearest neighbor concept
        idx_desc1 = np.argmin(dmat, axis=0)  # P positions selected from desc1
        # FIXME: case min distance links to more than one point
        # N-dimensional bidirectional selections array
        bidirectional_selection = np.arange(N) == idx_desc1[idx_desc2]
        selection = selection * bidirectional_selection  # N-dimensional array containing S True values

        # Computing 3xS matches matrix
        matches = np.vstack([
            np.arange(N),  # N original idxs from desc 1
            idx_desc2,  # N positions selected from desc2
            scores,  # N distance scores
        ])[:, selection]

        return matches

    def delete_oldest_points(self):
        """ deletes tracks column 3 (zero based) """
        if self.tracks.shape[1] >= 4:
            self.tracks = np.delete(self.tracks, 3, axis=1)

    def delete_empty_tracks_rows(self):
        """ deletes empty tracks rows """
        # computing the product of the tracks rows
        # FIXED
        tracks_rows_prod = self.tracks[:, 3:].max(axis=1)
        # deleting rows which are not tracking any point/descriptor
        self.tracks = np.delete(self.tracks, np.where(tracks_rows_prod == -1), axis=0)

    @classmethod
    def compute_cumulative_avg(
            cls, new_values: np.ndarray, previous_num_processed_values: np.ndarray, previous_ca: np.ndarray
    ) -> np.ndarray:
        r"""
        Computes the cumulative average (CA) as decribed below:
        https://en.wikipedia.org/wiki/Moving_average#Cumulative_average

        CA_{n+1} = CA_n + \frac{x_{n+1} - CA_n}{n + 1}

        Where:
            CA_n   : previous_ca
            x_{n+1}: new_values
            n      : previous_num_processed_values

        Kwargs:
            new_values             <np.ndarray>: 1D Array of new M scores to be processed. Where new_values_i
                                                 is the L2 distance between two descriptors.
            previous_num_processed_values <np.ndarray>: 1D array containing the number of values processed
                                                 so far.
            previous_ca            <np.ndarray>: 1D Array of M previous CA scores
        """
        assert isinstance(new_values, np.ndarray), type(new_values)
        assert len(new_values.shape) == 1, new_values.shape
        assert isinstance(previous_num_processed_values, np.ndarray), type(previous_num_processed_values)
        assert len(previous_num_processed_values.shape) == 1, previous_num_processed_values.shape
        assert previous_num_processed_values.min() >= 0, \
            'previous_num_processed_values cannot contain negative values'
        assert isinstance(previous_ca, np.ndarray), type(previous_ca)
        assert len(previous_ca.shape) == 1, previous_ca.shape

        return previous_ca + (new_values - previous_ca)/(previous_num_processed_values + 1)

    def update_ca_scores(self, scores: np.ndarray, idxs_to_update: list):
        """
        computes and updates tracks cumulative_avg (column 1) and num_desc (column 2) using provided
        indexeds

        Kwargs:
            scores   <np.ndarray>: 1D array of corresponding descriptors scores
            idxs_to_update <list>: list of tracks indexes to update with the scores
        """
        assert isinstance(scores, np.ndarray), type(scores)
        assert len(scores.shape) == 1, scores.shape
        assert 0 < scores.size <= self.tracks.shape[0], scores.size
        assert isinstance(idxs_to_update, list), type(idxs_to_update)
        assert len(idxs_to_update) == scores.size, (len(idxs_to_update), scores.size)

        # increasing counters of processed descriptors (num_desc)
        self.tracks[idxs_to_update, 2] += 1
        # computing cumulative average n + 1
        new_cumulative_avg = self.compute_cumulative_avg(
            scores, self.tracks[idxs_to_update, 2], self.tracks[idxs_to_update, 1])
        # updating cumulative averages
        self.tracks[idxs_to_update, 1] = new_cumulative_avg

    def get_new_column_for_appending(self, matches: np.ndarray) -> tuple:
        """
        Processes the matched decriptors indexes to create the new column to be added to tracks matrix

        Kwargs:
            matches <np.ndarray>: 2xM array of M matched descriptor positions from descriptor 1 and 2, where
                                  each colum is [idx_desc1_0_m, idx_desc2_1_m]^T

        Returns:
            new_tracks_col <np.ndarray>, idxs_updated <list>

        """
        assert isinstance(matches, np.ndarray), type(matches)
        assert len(matches.shape) == 2, matches.shape
        assert matches.shape[0] == 2, matches.shape[0]
        assert matches.shape[1] <= self.tracks.shape[0]

        new_tracks_col = np.full(self.tracks.shape[0], -1)  # initial new column values
        idxs_updated = []  # initial list of updated indexes

        # if matches is not empty and there is any match
        if matches.shape[1] > 0 and matches[1].max() != -1:
            # mapping last tracked descriptor 1 indexes to efficiently retrieve their positions from
            # tracks matrix
            val_to_idx = self.to_hashmap_by_value(self.tracks[:, -1])

            # updating new_tracks_col with matched indexes from descriptor 2
            for match_ in matches.T:
                idx = val_to_idx[match_[0]]  # retrieving tracks row idx to update
                new_tracks_col[idx] = match_[1]  # updating value from new_tracks_col
                idxs_updated.append(idx)  # storing idx updated

        return new_tracks_col, idxs_updated

    def append_matched_tracks(self, matches: np.ndarray, scores: np.ndarray):
        """
        Appends matched idxs as the last tracks column and updates their cumulative_avg and num_desc

        Kwargs:
            matches <np.ndarray>: 2xM array of M matched descriptor positions from descriptor 1 and 2, where
                                  each colum is [idx_desc1_0_m, idx_desc2_1_m]^T
            scores  <np.ndarray>: 1D array of corresponding M descriptors scores
        """
        assert isinstance(matches, np.ndarray), type(matches)
        assert isinstance(scores, np.ndarray), type(scores)
        assert len(matches.shape) == 2, matches.shape
        assert len(scores.shape) == 1, scores.shape
        assert matches.shape[1] <= self.tracks.shape[0], (matches.shape[1], self.tracks.shape[0])
        assert matches.shape[1] == scores.size, (matches.shape[1], scores.size)

        new_column, idxs_updated = self.get_new_column_for_appending(matches)

        # Appending descriptors indexes as the last column
        self.tracks = np.hstack([self.tracks, new_column[:, np.newaxis]])

        # if matches is not empty and there is any match
        if matches.shape[1] > 0 and matches[1].max() != -1:
            self.update_ca_scores(scores, idxs_updated)  # Updating cumulative_avg and num_desc

    def append_unmatched_tracks(self, idxs: np.ndarray = None):
        """
        appends unmathed idxs as new tracks rows

        Kwargs:
            idxs   <np.ndarray, None>: 1D array of descriptors positions.
                                       Default None
        """
        idxs = idxs if idxs is not None else np.empty(0)
        assert isinstance(idxs, np.ndarray), type(idxs)
        assert len(idxs.shape) == 1, idxs.shape

        if idxs.size > 0:
            new_tracks = np.full((idxs.size, self.tracks.shape[1]), -1)
            # setting tracking IDs
            new_tracks[:, 0] = np.arange(idxs.shape[0]) + self.total_tracks
            # setting initial cumulative_avg distance
            new_tracks[:, 1] = 0
            # setting initial num_desc
            new_tracks[:, 2] = 0
            # setting descriptor indexes (new unmatched tracks)
            new_tracks[:, -1] = idxs
            self.tracks = np.vstack([self.tracks, new_tracks])
            # increasing all-time tracked descriptors/points
            self.total_tracks += idxs.shape[0]

    def append_tracks(self, /, *, matched: np.ndarray = None, unmatched: np.ndarray = None):
        """
        Appends matched and unmatched tracks appropriately

        Kwargs:
            matched   <np.ndarray>: array [3, M] containing M columns of matched descriptors idxs and their
                                    corresponding scores [idx_desc1_0_m, idx_desc2_1_m score_2_m]^T
                                    Default None
            unmatched <np.ndarray>: 1D array containing unmatched descriptor idxs freom desc2
                                    Default None
        """
        assert matched is not None or unmatched is not None

        if matched is not None:
            assert isinstance(matched, np.ndarray), type(matched)
            assert len(matched.shape) == 2, 'matched must be a 2D array'
            assert matched.shape[0] == 3, 'matched must be a 3xM array'
        else:
            matched = np.vstack([
                np.full((2, self.tracks.shape[0]), -1, dtype=int),
                np.zeros(self.tracks.shape[0], dtype=int)
            ])

        if unmatched is not None:
            assert isinstance(unmatched, np.ndarray), type(unmatched)
            assert len(unmatched.shape) == 1, 'unmatched must be a 1D array'
        else:
            unmatched = np.empty(0, dtype=int)

        self.append_matched_tracks(matched[:2], matched[2])
        self.append_unmatched_tracks(unmatched)

    @staticmethod
    def to_hashmap_by_value(array: np.ndarray) -> dict:
        """
        Indexes a 1D numpy array by value using a dict

        Kwargs:
            array <np.ndarray>: 1D numpy array

        Returns:
            hashmap <dict>
        """
        assert isinstance(array, np.ndarray), type(array)
        assert len(array.shape) == 1, array.shape

        hashmap = {val: idx for idx, val in enumerate(array)}

        return hashmap

    def update(self, pts: np.ndarray, desc: np.ndarray):
        """
        Adds points and descriptors to tracks

        Kwargs:
            pts  <np.ndarray>: 3xN array of N points [x_0_n, y_1_n, confidence_2_n]
            desc <np.ndarray>: DxN array of corresponding N D-dimensional descriptors
        """
        assert isinstance(pts, np.ndarray), type(pts)
        assert isinstance(desc, np.ndarray), type(desc)
        assert pts.size > 0, pts.size
        assert desc.size > 0, desc.size
        assert pts.shape[1] == desc.shape[1], (f'pts.shape[1]={pts.shape[1]} must be equals to '
                                               f'desc.shape[1]={desc.shape[1]}')

        self.delete_oldest_points()
        self.delete_empty_tracks_rows()

        if self.prev_desc is None:  # first time adding points/descriptors
            matched = None
            unmatched = np.arange(pts.shape[1])
        else:
            matched = self.nn_match_two_way(self.prev_desc, desc, self.desc_threshold)
            unmatched = np.delete(np.arange(pts.shape[1]), matched[1].astype(int))

        self.append_tracks(matched=matched, unmatched=unmatched)
        self.prev_desc = desc

    def clear_desc(self):
        """ deletes prev_desc """
        # employed in export.py
        del self.prev_desc
        self.prev_desc = None
