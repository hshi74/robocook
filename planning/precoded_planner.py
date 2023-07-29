import copy
import numpy as np
import open3d as o3d
import subprocess
import torch
import yaml

from control_utils import *
from perception.pcd_utils import *
from planner import Planner
from utils.loss import chamfer
from utils.visualize import *


class PrecodedPlanner(Planner):
    def __init__(self, args):
        self.args = args

    def plan(self, state_cur, target_shape, rollout_path):
        super().plan(state_cur, target_shape, rollout_path)

    def eval_soln(self, param_seq, state_cur, state_end, target_shape):
        # state_cur and state_end are numpy array, both with size [1, 300, 3]
        state_end_norm, state_goal_norm = normalize_state(
            self.args, state_end[0], target_shape["surf"]
        )
        chamfer_loss = chamfer(state_end_norm, state_goal_norm)

        state_seq = np.concatenate((state_cur, state_end))
        state_seq_aug = np.repeat(state_seq, [20, 1], axis=0)

        title = f"{self.args.env.upper()}: {[round(x.item(), 3) for x in param_seq]}"
        title += f" -> {round(chamfer_loss, 4)}"

        pkl_path = os.path.join(
            self.rollout_path, "anim_args", f"{self.args.env}_anim_args.pkl"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "args": self.args,
                    "row_titles": [title],
                    "state_seqs": [state_seq_aug],
                    "target": target_shape["surf"],
                },
                f,
            )

        p = subprocess.Popen(
            ["python", "utils/visualize.py", pkl_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        with open(
            os.path.join(
                self.rollout_path, "param_seqs", f"{self.args.env}_param_seq.yml"
            ),
            "w",
        ) as f:
            yaml.dump({self.args.env: param_seq.tolist()}, f, default_flow_style=True)

        info_dict = {
            "loss": [chamfer_loss],
            "subprocess": [p],
        }

        return state_seq, info_dict


class CutterPlanarPlanner(PrecodedPlanner):
    def __init__(self, args):
        super().__init__(args)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)

        target_points = target_shape["surf"]

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        # target_mesh = alpha_shape_mesh_reconstruct(target_pcd, alpha=0.2, mesh_fix=True, visualize=False)
        # target_volume = target_mesh.volume
        target_mesh, _ = target_pcd.compute_convex_hull()
        target_volume = target_mesh.get_volume()
        # hard code
        target_volume = 2.7e-5
        print(f"The target volume is {target_volume}!")

        # visualize_o3d([target_pcd, target_hull], title='target_point_cloud')

        state_cur_pcd = o3d.geometry.PointCloud()
        state_cur_pcd.points = o3d.utility.Vector3dVector(
            state_cur.squeeze().cpu().numpy()
        )

        x_min, y_min, z_min = state_cur_pcd.get_min_bound()
        x_max, y_max, z_max = state_cur_pcd.get_max_bound()
        x_center, y_center, z_center = state_cur_pcd.get_center()

        cut_y = y_center
        cut_y_min = y_min
        cut_y_max = y_max
        volume_diff = 0
        last_volume_diff = float("inf")
        # binary search to find the right volume
        while abs(volume_diff - last_volume_diff) > 1e-9:
            last_volume_diff = volume_diff
            crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
                np.array([x_min, y_min, z_min]), np.array([x_max, cut_y, z_max])
            )
            state_cur_pcd_crop = copy.deepcopy(state_cur_pcd).crop(crop_bbox)
            # state_cur_mesh = alpha_shape_mesh_reconstruct(state_cur_pcd_crop, alpha=0.2, mesh_fix=True, visualize=False)
            # volume_cur = state_cur_mesh.volume
            state_cur_mesh, _ = state_cur_pcd_crop.compute_convex_hull()
            volume_cur = state_cur_mesh.get_volume()
            # visualize_o3d([state_cur_pcd_crop, state_cur_mesh], title='target_point_cloud')
            volume_diff = volume_cur - target_volume
            if volume_diff > 0:
                cut_y_max = cut_y
                cut_y = (cut_y + cut_y_min) / 2
            else:
                cut_y_min = cut_y
                cut_y = (cut_y + cut_y_max) / 2

            # print(volume_diff)

        # -0.008 because the geometry of the knife is not symmetric
        param_seq = torch.tensor(
            [x_center, cut_y - 0.008, np.pi / 4], dtype=torch.float32
        ).unsqueeze(0)

        return param_seq

    def eval_soln(self, param_seq, state_cur, target_shape):
        param_seq = param_seq.squeeze()
        state_cur_pcd = o3d.geometry.PointCloud()
        state_cur_pcd.points = o3d.utility.Vector3dVector(
            state_cur.squeeze().cpu().numpy()
        )

        x_min, y_min, z_min = state_cur_pcd.get_min_bound()
        x_max, y_max, z_max = state_cur_pcd.get_max_bound()

        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.array([x_min, y_min, z_min]),
            np.array([x_max, param_seq[1].item(), z_max]),
        )
        state_cur_pcd_crop = copy.deepcopy(state_cur_pcd).crop(crop_bbox)
        cropped_mesh = alpha_shape_mesh_reconstruct(
            state_cur_pcd_crop, alpha=0.2, mesh_fix=False
        )
        if self.args.surface_sample:
            state_end_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                cropped_mesh, state_cur.squeeze().shape[0]
            )
        else:
            raise NotImplementedError
        state_end = np.asarray(state_end_pcd.points)[None, :, :]

        return super().eval_soln(
            param_seq, state_cur.cpu().numpy(), state_end, target_shape
        )


class CutterCircularPlanner(PrecodedPlanner):
    def __init__(self, args):
        super().__init__(args)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)
        return self.center[:2].unsqueeze(0)

    def eval_soln(self, param_seq, state_cur, target_shape):
        state_cur_numpy = state_cur.cpu().numpy()
        return super().eval_soln(
            param_seq.squeeze(), state_cur_numpy, state_cur_numpy, target_shape
        )


class PusherPlanner(PrecodedPlanner):
    def __init__(self, args):
        super().__init__(args)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)
        return self.center[:2].unsqueeze(0)

    def eval_soln(self, param_seq, state_cur, target_shape):
        # radius of the circular cutter
        r = 0.035
        param_seq = param_seq.squeeze()
        state_cur_numpy = state_cur.squeeze().cpu().numpy()
        state_cur_crop_idx = (
            np.linalg.norm(
                state_cur_numpy[:, :2]
                - np.tile(param_seq.cpu().numpy()[:2], (state_cur_numpy.shape[0], 1)),
                axis=1,
            )
            < r
        )
        state_cur_crop = state_cur_numpy[state_cur_crop_idx]

        state_cur_pcd_crop = o3d.geometry.PointCloud()
        state_cur_pcd_crop.points = o3d.utility.Vector3dVector(state_cur_crop)

        cropped_mesh = alpha_shape_mesh_reconstruct(
            state_cur_pcd_crop, alpha=0.2, mesh_fix=False
        )
        if self.args.surface_sample:
            state_end_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                cropped_mesh, state_cur_numpy.shape[0]
            )
        else:
            raise NotImplementedError
        state_end = np.asarray(state_end_pcd.points)[None, :, :]

        return super().eval_soln(
            param_seq, state_cur.cpu().numpy(), state_end, target_shape
        )


class SpatulaPlanner(PrecodedPlanner):
    def __init__(self, args):
        super().__init__(args)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)
        if "small" in self.args.env:
            return torch.tensor(
                [self.center[0].item(), self.center[1].item(), 0.395, -0.29],
                dtype=torch.float32,
            ).unsqueeze(0)
        else:
            return torch.tensor(
                [0.5, -0.3, 0.395, -0.29], dtype=torch.float32
            ).unsqueeze(0)

    def eval_soln(self, param_seq, state_cur, target_shape):
        param_seq = param_seq.squeeze()
        state_cur_numpy = state_cur.squeeze().cpu().numpy()
        if "small" in self.args.env:
            state_trans = (
                np.array([param_seq[2], param_seq[3], 0.025])
                - self.center.cpu().numpy()
            )
            state_end = state_cur_numpy + np.tile(
                state_trans, (state_cur_numpy.shape[0], 1)
            )
            state_end = state_end[None, :, :]
        else:
            state_end = state_cur_numpy[None, :, :]

        return super().eval_soln(
            param_seq, state_cur.cpu().numpy(), state_end, target_shape
        )


class HookPlanner(PrecodedPlanner):
    def __init__(self, args):
        super().__init__(args)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)
        return self.center[:2].unsqueeze(0)

    def eval_soln(self, param_seq, state_cur, target_shape):
        return super().eval_soln(
            param_seq.squeeze(),
            state_cur.cpu().numpy(),
            target_shape["surf"][None, :, :],
            target_shape,
        )
