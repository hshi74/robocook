import copy
import glob
import os
import numpy as np
import torch
import torch.nn as nn

from pcd_tool_classifier import pointnet2_tool_cls
from utils.data_utils import *
from utils.visualize import *


class PcdClassifer(object):
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.device = args.device
        self.classes = sorted(args.tool_name_list)

        self.args.use_rgb = 1
        self.args.early_fusion = 0
        self.args.n_particles = 4096
        self.args.n_class = 15

        self.model = pointnet2_tool_cls.get_model(self.args)
        self.model.apply(inplace_relu)
        self.model = self.model.to(self.device)
        pretrained_dict = torch.load(
            self.args.tool_cls_model_path, map_location=self.device
        )
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        self.used_gripper = 0
        self.used_circular_cut = 0
        self.used_pusher = 0

    def eval(self, state_cur_dict, target_shape, path=""):
        state_pcd_list = [state_cur_dict["raw_pcd"], target_shape["raw_pcd"]]
        state_list = []
        for k in range(len(state_pcd_list)):
            state_pcd = state_pcd_list[k]
            state = np.concatenate(
                (np.asarray(state_pcd.points), np.asarray(state_pcd.colors)), axis=1
            )
            if not self.args.use_rgb:
                state = state[:, :3]
            if self.args.early_fusion:
                state = np.concatenate((state, np.full((state.shape[0], 1), k)), axis=1)
            state_list.append(state)
        state_concat = np.concatenate(state_list, axis=0)

        centroid = np.mean(state_list[0][:, :3], axis=0)
        state_concat[:, :3] = state_concat[:, :3] - centroid
        m = np.max(np.sqrt(np.sum(state_concat[:, :3] ** 2, axis=1)))
        state_concat[:, :3] = state_concat[:, :3] / m

        points = torch.tensor(
            state_concat, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        points = points.transpose(2, 1)
        output, _ = self.model(points)

        prob_output = output.softmax(dim=1).detach().cpu().numpy()[0]
        idx_pred = []
        for idx in prob_output.argsort():
            if prob_output[idx] > 0.1:
                idx_pred.insert(0, idx)
        # idx_pred = prob_output.argsort()[-1:]

        labels_pred = [self.classes[x] for x in idx_pred]

        visualize_pcd_pred(
            [f"IN: {labels_pred}", "OUT"],
            state_list,
            path=os.path.join(path, "cls.png"),
        )

        if not self.used_circular_cut:
            if (
                (
                    "cutter_circular" in labels_pred
                    and not "roller_large" in labels_pred
                    and not len(labels_pred) == 1
                )
                or "pusher" in labels_pred
                or "spatula_small" in labels_pred
            ):
                print(f"{labels_pred} triggered hard code!")
                labels_pred = ["cutter_circular"]

        if "cutter_circular" in labels_pred:
            if self.used_circular_cut > 0:
                print(f"{labels_pred} triggered hard code!")
                labels_pred = ["pusher"]
            else:
                self.used_circular_cut += 1

        if "pusher" in labels_pred:
            if self.used_pusher > 0:
                print(f"{labels_pred} triggered hard code!")
                labels_pred = ["spatula_small"]
            else:
                self.used_pusher += 1

        if not self.used_gripper:
            if "press_square" in labels_pred:
                print(f"{labels_pred} triggered hard code!")
                labels_pred = ["gripper_sym_plane"]

        if "gripper_sym_plane" in labels_pred:
            self.used_gripper += 1

        if len(labels_pred) == 1 and "roller_small" in labels_pred:
            labels_pred = ["roller_large"]

        # import pdb; pdb.set_trace()
        # labels_pred = ['hook']

        return labels_pred
