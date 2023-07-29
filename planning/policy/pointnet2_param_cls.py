import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from planning.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
)


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args

        in_channel = 0
        if args.use_normals:
            in_channel += 3
        if args.early_fusion:
            in_channel += 1

        if args.early_fusion:
            # npoint, radius_list, nsample_list, in_channel, mlp_list
            self.sa1 = PointNetSetAbstractionMsg(
                512,
                [0.1, 0.2, 0.4],
                [16, 32, 128],
                in_channel,
                [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2 = PointNetSetAbstractionMsg(
                128,
                [0.2, 0.4, 0.8],
                [32, 64, 128],
                320,
                [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            # npoint, radius, nsample, in_channel, mlp, group_all
            self.sa3 = PointNetSetAbstraction(
                None, None, None, 640 + 3, [256, 512, 1024], True
            )

            feature_channel = 1024
        else:
            self.sa1_in = PointNetSetAbstractionMsg(
                512,
                [0.1, 0.2, 0.4],
                [16, 32, 128],
                in_channel,
                [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2_in = PointNetSetAbstractionMsg(
                128,
                [0.2, 0.4, 0.8],
                [32, 64, 128],
                320,
                [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            self.sa3_in = PointNetSetAbstraction(
                None, None, None, 640 + 3, [256, 512, 1024], True
            )

            self.sa1_out = PointNetSetAbstractionMsg(
                512,
                [0.1, 0.2, 0.4],
                [16, 32, 128],
                in_channel,
                [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2_out = PointNetSetAbstractionMsg(
                128,
                [0.2, 0.4, 0.8],
                [32, 64, 128],
                320,
                [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            self.sa3_out = PointNetSetAbstraction(
                None, None, None, 640 + 3, [256, 512, 1024], True
            )

            feature_channel = 2048

        self.cls_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_channel, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_bin),
                    # nn.LogSoftmax(dim=-1)
                )
                for n_bin in args.n_bins
            ]
        )

        n_bins_aug = []
        for i in range(args.n_bins.shape[0]):
            if i % (args.n_bins.shape[0] // args.n_actions) == args.rot_idx:
                n_bins_aug.append(args.n_bins[i] * 2)
            else:
                n_bins_aug.append(args.n_bins[i])

        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_channel, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_bin),
                )
                for n_bin in n_bins_aug
            ]
        )

    def forward(self, xyz):
        B, _, n_points = xyz.shape

        if self.args.early_fusion:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        else:
            n_points_half = n_points // 2
            if self.args.use_normals:
                xyz_in = xyz[:, :3, :n_points_half]
                xyz_out = xyz[:, :3, n_points_half:]
                norm_in = xyz[:, 3:, :n_points_half]
                norm_out = xyz[:, 3:, n_points_half:]
            else:
                xyz_in = xyz[:, :, :n_points_half]
                xyz_out = xyz[:, :, n_points_half:]
                norm_in = None
                norm_out = None

            l1_xyz_in, l1_points_in = self.sa1_in(xyz_in, norm_in)
            l2_xyz_in, l2_points_in = self.sa2_in(l1_xyz_in, l1_points_in)
            l3_xyz_in, l3_points_in = self.sa3_in(l2_xyz_in, l2_points_in)
            l1_xyz_out, l1_points_out = self.sa1_out(xyz_out, norm_out)
            l2_xyz_out, l2_points_out = self.sa2_out(l1_xyz_out, l1_points_out)
            l3_xyz_out, l3_points_out = self.sa3_out(l2_xyz_out, l2_points_out)

            l3_points = torch.cat((l3_points_in, l3_points_out), dim=1)

        x = l3_points.view(B, l3_points.shape[1])

        pred_ind = []
        pred_offsets = []
        for i in range(len(self.args.n_bins)):
            pred_idx = self.cls_heads[i](x)
            pred_ind.append(pred_idx)

            pred_offset = self.reg_heads[i](x)
            if (
                i % (self.args.n_bins.shape[0] // self.args.n_actions)
                == self.args.rot_idx
            ):
                # sin and cos
                pred_offset = pred_offset.view([B, -1, 2])
                pred_offset_norm = F.normalize(pred_offset, dim=2)
                pred_angle = torch.atan2(
                    pred_offset_norm[:, :, 1], pred_offset_norm[:, :, 0]
                )
                half_bin_size = torch.tensor(
                    (
                        self.args.tool_params["rot_range"][1]
                        - self.args.tool_params["rot_range"][0]
                    )
                    / self.args.n_bins[i],
                    device=self.args.device,
                )
                pred_angle *= half_bin_size / np.pi
                pred_offsets.append(pred_angle)
            else:
                pred_offsets.append(pred_offset)

        return (pred_ind, pred_offsets), l3_points


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.args = args
        self.cls_loss_func = nn.CrossEntropyLoss()

        if args.reg_loss_type == "l1":
            self.reg_loss_func = nn.L1Loss()
        elif args.reg_loss_type == "smooth_l1":
            self.reg_loss_func = nn.SmoothL1Loss()
        elif args.reg_loss_type == "huber":
            self.reg_loss_func = nn.HuberLoss()
        elif args.reg_loss_type == "mse":
            self.reg_loss_func = nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, pred, target):
        target_idx, target_params = target
        pred_ind, pred_offsets = pred

        n_gt_bins = target_idx.shape[-1]
        cls_loss = 0.0
        reg_loss = 0.0
        for i in range(len(self.args.n_bins)):
            cls_loss += self.args.cls_weight * self.cls_loss_func(
                pred_ind[i], target_idx[:, i, 0]
            )
            for j in range(n_gt_bins):
                bin_idx_gt = target_idx[:, i, j]
                bin_centers_batch = []
                for idx in bin_idx_gt:
                    bin_centers_batch.append(self.args.bin_centers_torch[i][idx])
                bin_centers_batch = torch.stack(bin_centers_batch).to(bin_idx_gt.device)
                pred_offset = pred_offsets[i][
                    torch.arange(bin_idx_gt.shape[0]), bin_idx_gt
                ]
                if (
                    i % (self.args.n_bins.shape[0] // self.args.n_actions)
                    == self.args.rot_idx
                ):
                    period_ratio = np.pi * 2 / self.args.tool_params["rot_scope"]
                    reg_loss += (
                        -self.args.orient_weight
                        / n_gt_bins
                        * torch.cos(
                            (target_params[:, i] - bin_centers_batch - pred_offset)
                            * period_ratio
                        ).mean()
                    )
                else:
                    reg_loss += (
                        1
                        / n_gt_bins
                        * self.reg_loss_func(
                            bin_centers_batch + pred_offset, target_params[:, i]
                        )
                    )

        return cls_loss, reg_loss
