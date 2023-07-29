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
        if args.use_rgb:
            in_channel += 3
        if args.early_fusion:
            in_channel += 1

        if args.early_fusion:
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

        self.cls_head = nn.Sequential(
            nn.Linear(feature_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.args.n_class),
        )

    def forward(self, xyz):
        B, _, n_points = xyz.shape
        if self.args.early_fusion:
            if self.args.use_rgb:
                rgb = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                rgb = None
            l1_xyz, l1_points = self.sa1(xyz, rgb)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        else:
            n_points_half = n_points // 2
            if self.args.use_rgb:
                xyz_in = xyz[:, :3, :n_points_half]
                xyz_out = xyz[:, :3, n_points_half:]
                rgb_in = xyz[:, 3:, :n_points_half]
                rgb_out = xyz[:, 3:, n_points_half:]
            else:
                xyz_in = xyz[:, :, :n_points_half]
                xyz_out = xyz[:, :, n_points_half:]
                rgb_in = None
                rgb_out = None

            l1_xyz_in, l1_points_in = self.sa1_in(xyz_in, rgb_in)
            l2_xyz_in, l2_points_in = self.sa2_in(l1_xyz_in, l1_points_in)
            l3_xyz_in, l3_points_in = self.sa3_in(l2_xyz_in, l2_points_in)
            l1_xyz_out, l1_points_out = self.sa1_out(xyz_out, rgb_out)
            l2_xyz_out, l2_points_out = self.sa2_out(l1_xyz_out, l1_points_out)
            l3_xyz_out, l3_points_out = self.sa3_out(l2_xyz_out, l2_points_out)

            l3_points = torch.cat((l3_points_in, l3_points_out), dim=1)

        x = l3_points.view(B, l3_points.shape[1])

        pred = self.cls_head(x)

        return pred, l3_points


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.args = args
        self.cls_loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        cls_loss = self.cls_loss_func(pred, target)

        return cls_loss
