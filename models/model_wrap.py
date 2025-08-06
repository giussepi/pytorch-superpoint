# -*- coding: utf-8 -*-
""" models/model_wrap.py

class to process superpoint net
# may be some duplication with model_wrap.py
# PointTracker is from Daniel's repo.
"""

import numpy as np
import torch
from torch import nn, Tensor

from models.SuperPointNet_pretrained import SuperPointNet
from utils.loader import modelLoader
from utils.losses import extract_patch_from_points, soft_argmax_2d, norm_patches, do_log
from utils.utils import toNumpy, extract_points, flattenDetection


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


class SuperPointFrontend_torch(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    '''
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    '''

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

    def soft_argmax_points(self, pts, patch_size=5):
        """
        input:
            pts: tensor [N x 2]
        """
        ##### check not take care of batch #####
        # print("not take care of batch! only take first element!")
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(self.heatmap, pts, patch_size=patch_size)
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


class PointTracker(object):
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
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        self.matches = None
        self.last_pts = None
        self.mscores = None

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - MxN numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - MxN numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]

        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))

        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
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

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.

        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        # pts = pts.transpose()
        # desc = desc.transpose()
        assert pts.shape[1] == desc.shape[1]
        # assert pts.shape[0] == desc.shape[0]
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
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        self.matches = matches
        pts_id = pts[:2, :]
        if self.last_pts is not None:
            id1, id2 = self.last_pts[:, matches[0, :].astype(int)], pts_id[:, matches[1, :].astype(int)]

            self.matches = np.concatenate((id1, id2), axis=0)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
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
        self.last_pts = pts[:2, :].copy()

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()

        return returned_tracks
