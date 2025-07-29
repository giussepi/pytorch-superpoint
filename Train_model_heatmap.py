# -*- coding: utf-8 -*-
""" Train_model_heatmap.py
This is the main training interface using heatmap trick
Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import logging
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch import nn, Tensor

from Train_model_frontend import Train_model_frontend
from utils.d2s import DepthToSpace
from utils.dict_ops import dict_sum, dict_div_by_scalar, dict_only_primitives
from utils.loader import dataLoader
from utils.losses import do_log, extract_patches, soft_argmax_2d, norm_patches
from utils.tools import dict_update
from utils.utils import (
    flattenDetection, toNumpy, img_overlap, to_floatTensor, labels2Dto3D,
    getPtsFromHeatmap, precision, recall, f1_score, accuracy, balanced_accuracy)


__all__ = [
    'Train_model_heatmap',
]


class Train_model_heatmap(Train_model_frontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    DEFAULT_CONFIG = {
        "retrain": True,
        "reset_epoch_iter": True,
        "epochs": 100,
        "validations_per_epoch": 1,
        "tensorboard_epoch_interval": 1,
        "savings_per_epoch": 1,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # Update config
        print("Load Train_model_heatmap!!")
        self.config = self.DEFAULT_CONFIG
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self.cell_size = 8
        self.subpixel = False
        self.epochs = config["epochs"]
        self.current_epoch = 0
        self.best_epoch = 0  # Epoch when the best score is achieved
        self.best_score = -math.inf  # Best metrics average weighted sum after an epoch
        self.best_metrics = None  # Overall best metrics based on best_score
        # NOTE: all metrics must have the same signature
        self.metrics_fn = [precision, recall, f1_score, accuracy, balanced_accuracy]
        # used to compute the score to determine the best model
        self.metrics_weights = [0, 1, 0, 0, 0]
        self.metrics_keys = [_.__name__ for _ in self.metrics_fn]
        self.n_iter = 0
        self.net = None
        self.optimizer = None
        self.gaussian = False

        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        self.printImportantConfig()

    def detector_loss(self, pred, target, mask=None, loss_type="softmax"):
        """
        apply loss on detectors, default is softmax

        :param pred: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(pred, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            loss = loss_func_BCE(nn.functional.softmax(pred, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    @staticmethod
    def update_overlap(images_dict, labels_warp_2D, heatmap_nms_batch, img_warp, name):
        nms_overlap = [
            img_overlap(
                toNumpy(labels_warp_2D[i]),
                heatmap_nms_batch[i],
                toNumpy(img_warp[i]),
            )
            for i in range(heatmap_nms_batch.shape[0])
        ]
        nms_overlap = np.stack(nms_overlap, axis=0)
        images_dict.update({name + "_nms_overlap": nms_overlap})

    def compute_score(self, data: dict) -> float:
        """
        Computes the overall score considering the metrics weights

        Kwargs:
            data <dict>: dictionary containing averaged scores

        Returns:
            score <float>
        """
        assert isinstance(data, dict), type(data)

        score = sum(data[k]*w for k, w in zip(self.metrics_keys, self.metrics_weights))
        score /= sum(self.metrics_weights)

        return score

    def train_val_sample(self, sample, tb_interval, running_data, n_iter=0, train=False):
        """
        # key function
        :param sample:
        :param tb_interval:
        :param running_data:
        :param n_iter:
        :param train:
        :return:
        """
        task = "train" if train else "val"
        self.net.train(train)  # when train = False, it works like self.net.eval()
        if_warp = self.config['data']['warped_pair']['enable']

        scalar_dict, images_dict, hist_dict = {}, {}, {}
        # get the inputs
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_labels"],
                sample["warped_valid_mask"],
            )

        # homographies
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self.net(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
        else:
            with torch.no_grad():
                outs = self.net(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]

        # detector loss
        warped_labels = None
        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
            if if_warp:
                warped_labels = sample["warped_labels_gaussian"]
        else:
            labels_2D = sample["labels_2D"]
            if if_warp:
                warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        labels_3D = labels2Dto3D(
            labels_2D.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin
        ).float()
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.detector_loss(
            pred=outs["semi"],
            target=labels_3D.to(self.device),
            mask=mask_3D_flattened,
            loss_type=det_loss_type,
        )
        # warp
        if if_warp:
            assert warped_labels is not None
            labels_3D = labels2Dto3D(
                warped_labels.to(self.device),
                cell_size=self.cell_size,
                add_dustbin=add_dustbin,
            ).float()
            mask_3D_flattened = self.getMasks(
                mask_warp_2D, self.cell_size, device=self.device
            )
            loss_det_warp = self.detector_loss(
                pred=outs_warp["semi"],
                target=labels_3D.to(self.device),
                mask=mask_3D_flattened,
                loss_type=det_loss_type,
            )
        else:
            loss_det_warp = torch.tensor([0]).float().to(self.device)

        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                mask_valid=mask_desc,
                device=self.device,
                **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        ##### try to minimize the error ######
        # the following code is never called
        # add_res_loss = False
        # if add_res_loss and n_iter % 10 == 0:
        #     print("add_res_loss!!!")
        #     heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
        #     heatmap_org_nms_batch = self.heatmap_to_nms(
        #         images_dict, heatmap_org, name="heatmap_org"
        #     )
        #     if if_warp:
        #         heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
        #         heatmap_warp_nms_batch = self.heatmap_to_nms(
        #             images_dict, heatmap_warp, name="heatmap_warp"
        #         )

        #     # original: pred
        #     # check the loss on given labels!
        #     outs_res = self.get_residual_loss(
        #         sample["labels_2D"]
        #         * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
        #         heatmap_org,
        #         sample["labels_res"],
        #         scalar_dict,
        #         images_dict,
        #         hist_dict,
        #         name="original_pred",
        #     )
        #     loss_res_ori = (outs_res["loss"] ** 2).mean()
        #     # warped: pred
        #     if if_warp:
        #         outs_res_warp = self.get_residual_loss(
        #             sample["warped_labels"]
        #             * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
        #             heatmap_warp,
        #             sample["warped_res"],
        #             scalar_dict,
        #             images_dict,
        #             hist_dict,
        #             name="warped_pred",
        #         )
        #         loss_res_warp = (outs_res_warp["loss"] ** 2).mean()
        #     else:
        #         loss_res_warp = torch.tensor([0]).to(self.device)
        #     loss_res = loss_res_ori + loss_res_warp
        #     # print("loss_res requires_grad: ", loss_res.requires_grad)
        #     loss += loss_res
        #     scalar_dict.update(
        #         {"loss_res_ori": loss_res_ori, "loss_res_warp": loss_res_warp}
        #     )

        self.loss = loss
        scalar_dict.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )
        self.input_to_imgDict(sample, images_dict)

        if train:
            loss.backward()
            self.optimizer.step()

        # add clean map to tensorboard
        # semi_warp: flatten, to_numpy
        heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
        heatmap_org_nms_batch = self.heatmap_to_nms(
            images_dict, heatmap_org, name="heatmap_org"
        )
        if if_warp:
            heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
            heatmap_warp_nms_batch = self.heatmap_to_nms(
                images_dict, heatmap_warp, name="heatmap_warp"
            )

        self.update_overlap(
            images_dict,
            labels_2D,
            heatmap_org_nms_batch[np.newaxis, ...],
            img,
            "original",
        )

        self.update_overlap(
            images_dict,
            labels_2D,
            toNumpy(heatmap_org),
            img,
            "original_heatmap",
        )
        if if_warp:
            self.update_overlap(
                images_dict,
                labels_warp_2D,
                heatmap_warp_nms_batch[np.newaxis, ...],
                img_warp,
                "warped",
            )
            self.update_overlap(
                images_dict,
                labels_warp_2D,
                toNumpy(heatmap_warp),
                img_warp,
                "warped_heatmap",
            )
        # residuals
        if self.gaussian:
            # original: gt
            self.get_residual_loss(
                sample["labels_2D"],
                sample["labels_2D_gaussian"],
                sample["labels_res"],
                scalar_dict,
                images_dict,
                hist_dict,
                name="original_gt",
            )
            if if_warp:
                # warped: gt
                self.get_residual_loss(
                    sample["warped_labels"],
                    sample["warped_labels_gaussian"],
                    sample["warped_res"],
                    scalar_dict,
                    images_dict,
                    hist_dict,
                    name="warped_gt",
                )

        metrics = self.compute_metrics(
            to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
            sample["labels_2D"],
        )
        scalar_dict.update(metrics)
        running_data[task] = dict_sum(running_data[task], dict_only_primitives(scalar_dict))

        if (task == "train" and (n_iter % tb_interval == 0)) or (task == "val" and (n_iter == self.n_iter)):
            logging.info("%s current iteration: %d", task, n_iter)
            running_data[task] = dict_div_by_scalar(running_data[task], tb_interval)  # data per batch
            running_data[task]['overall_score'] = self.compute_score(running_data[task])

            if task == 'val' and running_data[task]['overall_score'] > self.best_score:
                self.best_score = running_data[task]['overall_score']
                self.best_metrics = deepcopy(running_data[task])
                self.best_epoch = self.current_epoch
                logging.info(
                    "Best overall score of %.4f achieved at epoch %d iteration %d",
                    self.best_score, self.best_epoch, self.n_iter
                )
                self.save_best_model()

            self.printLosses(running_data[task], task)
            # self.tb_images_dict(task, images_dict, max_img=2)
            # self.tb_hist_dict(task, hist_dict)
            self.tb_scalar_dict(running_data[task], task)
            running_data[task].clear()

        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        heatmap_np = toNumpy(heatmap)
        # heatmap_nms
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    def get_residual_loss(
            self, labels_2D, heatmap, labels_res, scalar_dict, images_dict, hist_dict, name=""):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self.device
        )
        hist_dict[name + "_resi_loss_x"] = outs_res["loss"][:, 0]
        hist_dict[name + "_resi_loss_y"] = outs_res["loss"][:, 1]
        err = abs(outs_res["loss"]).mean(dim=0)
        # print("err[0]: ", err[0])
        var = abs(outs_res["loss"]).std(dim=0)
        scalar_dict[name + "_resi_loss_x"] = err[0]
        scalar_dict[name + "_resi_loss_y"] = err[1]
        scalar_dict[name + "_resi_var_x"] = var[0]
        scalar_dict[name + "_resi_var_y"] = var[1]
        images_dict[name + "_patches"] = outs_res["patches"]

        return outs_res

    def compute_metrics(self, preds: Tensor, labels: Tensor) -> dict[float]:
        """
        Computes and returns the metrics defined in __init__ -> self.metrics_fn

        Kwargs:
            pred   <Tensor>: binary tensor [B, C, H, W]
            labels <Tensor>: binary tensor [B, C, H, W]

        Returns:
            metrics <dict>
        """
        metrics = {}

        for metric_fn in self.metrics_fn:
            metrics.update(metric_fn(preds, labels))

        return metrics

    @staticmethod
    def ext_from_points(labels_res, points):
        """
        Extracts residual

        input:
            labels_res: tensor [batch, channel, H, W]
            points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
        return:
            tensor [N, channel]
        """
        labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
        points_res = labels_res[
            points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
        ]  # tensor [N, 2]

        return points_res

    @classmethod
    def pred_soft_argmax(cls, labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        outs = {}
        # extract patches
        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # Extracts residual
        points_res = cls.ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches

        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input:
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)

        return heatmap

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)

        return heatmap

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        heatmap = heatmap.squeeze()
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[
            pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)
        ] = 1

        return semi_thd_nms_sample


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_train_heatmap.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_heatmap(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # try:
    #     model_fe.train()

    # except KeyboardInterrupt:
    #     logging.info("ctrl + c is pressed. save model")
