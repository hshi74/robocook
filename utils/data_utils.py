import copy
import h5py
import numpy as np
import open3d as o3d
import os
import random
import sys
import torch

from perception.pcd_utils import fps
from transforms3d.axangles import axangle2mat
from utils.visualize import *


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


def store_data(data_names, data, path):
    hf = h5py.File(path, "w")
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, "r")
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def expand(batch_size, info):
    length = len(info.shape)
    if length == 2:
        info = info.expand([batch_size, -1])
    elif length == 3:
        info = info.expand([batch_size, -1, -1])
    elif length == 4:
        info = info.expand([batch_size, -1, -1, -1])
    return info


def get_scene_info(data):
    """
    A subset of prepare_input() just to get number of particles
    for initialization of grouping
    """
    positions, shape_quats, scene_params = data
    n_shapes = shape_quats.shape[0]
    count_nodes = positions.shape[0]
    n_particles = count_nodes - n_shapes

    return n_particles, n_shapes, scene_params


def normalize_scene_param(scene_params, param_idx, param_range, norm_range=(-1, 1)):
    normalized = scene_params[:, param_idx]
    low, high = param_range
    if low == high:
        return normalized
    nlow, nhigh = norm_range
    normalized = nlow + (normalized - low) * (nhigh - nlow) / (high - low)
    return normalized


# @profile
def get_env_group(args, B):
    p_rigid = torch.zeros(B, args.n_instance, device=args.device)
    p_rigid[:] = args.p_rigid

    p_instance = torch.zeros(B, args.n_particles, args.n_instance, device=args.device)
    for i in range(args.n_instance):
        p_instance[:, :, i] = 1

    physics_param = torch.zeros(B, args.n_particles, device=args.device)
    scene_params = torch.tensor(
        args.scene_params, device=args.device, dtype=torch.float32
    )
    scene_params = torch.tile(scene_params, (B, 1))
    norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
    physics_param[:] = norm_g.float().view(B, 1)

    # p_rigid: B x n_instance
    # p_instance: B x n_p x n_instance
    # physics_param: B x n_p
    return [p_rigid, p_instance, physics_param]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def batch_normalize(batch, mean, std, eps=1e-10):
    if len(mean.shape) == 1:
        return (batch - mean) / (std + eps)
    elif len(mean.shape) == 2:
        batch_new = []
        for i in range(batch.shape[0]):
            batch_new.append((batch[i] - mean[i]) / (std[i] + eps))
        return torch.stack(batch_new)
    else:
        raise NotImplementedError


def batch_denormalize(batch, mean, std, eps=1e-10):
    if len(mean.shape) == 1:
        return batch * (std + eps) + mean
    elif len(mean.shape) == 2:
        batch_new = []
        for i in range(batch.shape[0]):
            batch_new.append(batch[i] * (std[i] + eps) + mean[i])
        return torch.stack(batch_new)
    else:
        raise NotImplementedError


def get_circle(center, radius, dim, axis, alpha=1.5):
    # sunflower seed arrangement
    # https://stackoverflow.com/a/72226595
    n_exterior = np.round(alpha * np.sqrt(dim)).astype(int)
    n_interior = dim - n_exterior

    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * dim, dim)

    r_interior = np.linspace(0, 1, n_interior)
    r_exterior = np.ones(n_exterior)
    r = radius * np.concatenate((r_interior, r_exterior))

    circle_2d = r * np.stack((np.cos(angles), np.sin(angles)))
    circle = np.concatenate((circle_2d[:axis], np.zeros((1, dim)), circle_2d[axis:]))
    circle = circle.T + center

    return circle


def get_square(center, unit_size, dim, axis):
    state = []
    n_rows = int(np.sqrt(dim))
    for i in range(dim):
        row = i // n_rows - (n_rows - 1) / 2
        col = i % n_rows - (n_rows - 1) / 2
        pos = [unit_size * row, unit_size * col]
        pos.insert(axis, 0)
        state.append(center + np.array(pos))
    return np.array(state, dtype=np.float32)


def load_tools(args):
    tool_full_repr_dict = {}
    for tool_name, tool_geom_list in args.tool_geom_mapping.items():
        tool_repr_points_list = []
        for i in range(len(tool_geom_list)):
            tool_repr_points_path = os.path.join(
                args.tool_repr_path, f"{tool_geom_list[i]}_points.npy"
            )
            if os.path.exists(tool_repr_points_path):
                tool_repr_points_list.append(
                    np.load(tool_repr_points_path, allow_pickle=True)
                )
            else:
                tool_mesh = o3d.io.read_triangle_mesh(
                    os.path.join(args.tool_repr_path, f"{tool_geom_list[i]}.stl")
                )
                tool_surface_dense = o3d.geometry.TriangleMesh.sample_points_uniformly(
                    tool_mesh, 100000, seed=0
                )

                if "press_circle" in tool_geom_list[i]:
                    voxel_size = 0.0057
                else:
                    voxel_size = 0.006

                tool_surface = tool_surface_dense.voxel_down_sample(
                    voxel_size=voxel_size
                )

                if "press_circle" in tool_geom_list[i]:
                    tool_repr_points = fps(
                        np.asarray(tool_surface.points),
                        args.tool_dim["press_circle"][i],
                    )
                else:
                    tool_repr_points = np.asarray(tool_surface.points)

                tool_repr_points_list.append(tool_repr_points)
                # with open(tool_repr_points_path, 'wb') as f:
                #     np.save(f, tool_repr_points)

        tool_full_repr_dict[tool_name] = tool_repr_points_list

    return tool_full_repr_dict


def get_normals(points, pkg="numpy"):
    B = points.shape[0]
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=8)
    if pkg == "torch":
        points = points.detach().cpu().numpy().astype(np.float64)

    tool_normals_list = []
    for b in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[b])
        pcd.estimate_normals(search_param, fast_normal_computation=True)
        pcd.orient_normals_towards_camera_location(pcd.get_center())
        normals = np.negative(np.asarray(pcd.normals))
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        # visualize_o3d([pcd], show_normal=True)
        if pkg == "torch":
            normals = torch.tensor(normals, dtype=torch.float32)

        tool_normals_list.append(normals)

    if pkg == "torch":
        return torch.stack(tool_normals_list)
    else:
        return np.stack(tool_normals_list)


def get_tool_repr(args, fingertip_T_list, pkg="numpy"):
    tool_dim_list = args.tool_dim[args.env]
    tool_center_list = args.tool_center[args.env]
    if args.full_repr:
        tool_repr_list = []
        for i in range(len(fingertip_T_list)):
            if pkg == "numpy":
                tool_repr_part = (
                    fingertip_T_list[i][:3, :3]
                    @ np.array(args.tool_full_repr_dict[args.env][i]).T
                ).T + np.tile(fingertip_T_list[i][:3, 3], (tool_dim_list[i], 1))
            else:
                tool_repr_part = (
                    fingertip_T_list[i][:3, :3]
                    @ torch.FloatTensor(args.tool_full_repr_dict[args.env][i]).T
                ).T + torch.tile(fingertip_T_list[i][:3, 3], (tool_dim_list[i], 1))
            tool_repr_list.append(tool_repr_part)

        if pkg == "numpy":
            tool_repr = np.concatenate(tool_repr_list)
        if pkg == "torch":
            tool_repr = torch.cat(tool_repr_list)
    else:
        if "gripper" in args.env:
            unit_size = 0.05 / (tool_dim_list[0] - 1)

            def get_gripper_repr(i, gripper_center):
                tool_repr = []
                for j in range(tool_dim_list[i]):
                    p = [gripper_center[args.axes[0]], gripper_center[args.axes[1]]]
                    p.insert(
                        args.axes[2],
                        gripper_center[args.axes[2]]
                        + unit_size * (j - (tool_dim_list[i] - 1) / 2),
                    )
                    tool_repr.append(p)

                if pkg == "numpy":
                    tool_repr = (
                        fingertip_T_list[i][:3, :3] @ np.array(tool_repr).T
                    ).T + np.tile(fingertip_T_list[i][:3, 3], (tool_dim_list[i], 1))
                else:
                    tool_repr = (
                        fingertip_T_list[i][:3, :3] @ torch.FloatTensor(tool_repr).T
                    ).T + torch.tile(fingertip_T_list[i][:3, 3], (tool_dim_list[i], 1))

                return tool_repr

            if "rod" in args.env or "asym" in args.env:
                gripper_center_l = tool_center_list[0]
                tool_repr_l = get_gripper_repr(0, gripper_center_l)
            else:
                gripper_center_l = tool_center_list[0]
                unit_size = 0.05 / (int(np.sqrt(tool_dim_list[0])) - 1)
                tool_repr_l = get_square(
                    gripper_center_l, unit_size, tool_dim_list[0], 0
                )

                if pkg == "numpy":
                    tool_repr_l = (
                        fingertip_T_list[0][:3, :3] @ tool_repr_l.T
                    ).T + np.tile(fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1))
                else:
                    tool_repr_l = (
                        fingertip_T_list[0][:3, :3] @ torch.FloatTensor(tool_repr_l).T
                    ).T + torch.tile(fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1))

            if "rod" in args.env:
                gripper_center_r = tool_center_list[1]
                tool_repr_r = get_gripper_repr(1, gripper_center_r)
            else:
                gripper_center_r = tool_center_list[1]
                unit_size = 0.05 / (int(np.sqrt(tool_dim_list[1])) - 1)
                tool_repr_r = get_square(
                    gripper_center_r, unit_size, tool_dim_list[1], 0
                )

                if pkg == "numpy":
                    tool_repr_r = (
                        fingertip_T_list[1][:3, :3] @ tool_repr_r.T
                    ).T + np.tile(fingertip_T_list[1][:3, 3], (tool_dim_list[1], 1))
                else:
                    tool_repr_r = (
                        fingertip_T_list[1][:3, :3] @ torch.FloatTensor(tool_repr_r).T
                    ).T + torch.tile(fingertip_T_list[1][:3, 3], (tool_dim_list[1], 1))

            if pkg == "numpy":
                tool_repr = np.concatenate((tool_repr_l, tool_repr_r))
            else:
                tool_repr = torch.cat((tool_repr_l, tool_repr_r))

        elif "roller" in args.env:
            unit_size = 0.07 / (tool_dim_list[0] - 1)
            tool_repr = []
            for j in range(tool_dim_list[0]):
                p = [
                    tool_center_list[0][args.axes[1]],
                    tool_center_list[0][args.axes[2]],
                ]
                p.insert(
                    args.axes[0],
                    tool_center_list[0][args.axes[0]]
                    + unit_size * (j - (tool_dim_list[0] - 1) / 2),
                )
                tool_repr.append(p)

            if pkg == "numpy":
                tool_repr = (
                    fingertip_T_list[0][:3, :3] @ np.array(tool_repr).T
                ).T + np.tile(fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1))
            else:
                tool_repr = (
                    fingertip_T_list[0][:3, :3] @ torch.FloatTensor(tool_repr).T
                ).T + torch.tile(fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1))

        elif "press" in args.env or "punch" in args.env:
            press_center = tool_center_list[0]
            if "square" in args.env:
                if "press" in args.env:
                    unit_size = 0.04 / (int(np.sqrt(tool_dim_list[0])) - 1)
                else:
                    unit_size = 0.018 / (int(np.sqrt(tool_dim_list[0])) - 1)
                tool_repr = get_square(press_center, unit_size, tool_dim_list[0], 2)
            else:
                radius = 0.02 if "press" in args.env else 0.01
                tool_repr = get_circle(press_center, radius, tool_dim_list[0], 2)

            if pkg == "numpy":
                tool_repr = (fingertip_T_list[0][:3, :3] @ tool_repr.T).T + np.tile(
                    fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1)
                )
            else:
                tool_repr = (
                    fingertip_T_list[0][:3, :3] @ torch.FloatTensor(tool_repr).T
                ).T + torch.tile(fingertip_T_list[0][:3, 3], (tool_dim_list[0], 1))
        else:
            raise NotImplementedError

    return tool_repr


def add_shape_to_seq(args, state_seq, init_pose_seq, act_seq):
    state_seq_new = []
    for i in range(act_seq.shape[0]):
        tool_pos_list = []
        tool_start = 0
        for j in range(len(args.tool_dim[args.env])):
            tool_dim = args.tool_dim[args.env][j]
            tool_pos = init_pose_seq[i, tool_start : tool_start + tool_dim]
            tool_pos_list.append(tool_pos)
            tool_start += tool_dim

        for j in range(act_seq.shape[1]):
            step = i * act_seq.shape[1] + j
            for k in range(len(args.tool_dim[args.env])):
                tool_pos_list[k] += np.tile(
                    act_seq[i, j, 6 * k : 6 * k + 3], (tool_pos_list[k].shape[0], 1)
                )
                act_rot = act_seq[i, j, 6 * k + 3 : 6 * k + 6]
                if args.full_repr and "roller" in args.env and np.any(act_rot):
                    act_trans = np.mean(tool_pos_list[k], axis=0)
                    tool_pos_list[k] -= np.tile(
                        act_trans, (tool_pos_list[k].shape[0], 1)
                    )
                    # rot_axis = [-act_seq[i, j, 6*k+1], act_seq[i, j, 6*k], 0]
                    # hard code...
                    if "large" in args.env:
                        rot_axis = -np.cross(
                            tool_pos_list[k][0] - tool_pos_list[k][1],
                            tool_pos_list[k][0] - tool_pos_list[k][2],
                        )
                    else:
                        if args.stage == "dy":
                            rot_axis = np.cross(
                                tool_pos_list[k][0] - tool_pos_list[k][1],
                                tool_pos_list[k][0] - tool_pos_list[k][2],
                            )
                        else:
                            rot_axis = np.cross(
                                tool_pos_list[k][0] - tool_pos_list[k][1],
                                tool_pos_list[k][0] - tool_pos_list[k][42],
                            )
                    tool_pos_list[k] = (
                        axangle2mat(rot_axis, act_rot[0]) @ tool_pos_list[k].T
                    ).T
                    tool_pos_list[k] += np.tile(
                        act_trans, (tool_pos_list[k].shape[0], 1)
                    )

            state_new = np.concatenate(
                [state_seq[step], args.floor_state, *tool_pos_list]
            )
            state_seq_new.append(state_new)

    return np.array(state_seq_new)


def get_act_seq_from_state_seq(args, state_shape_seq):
    # for rollers
    spread = False
    roller_motion_z_dist_prev = 0

    act_seq = []
    actions = []
    # state_diff_list = []
    for i in range(1, state_shape_seq.shape[0]):
        action = []
        tool_start = args.n_particles + args.floor_dim

        if args.full_repr and "roller" in args.env:
            roller_motion = np.mean(state_shape_seq[i, tool_start:], axis=0) - np.mean(
                state_shape_seq[i - 1, tool_start:], axis=0
            )
            roller_motion_z_dist = abs(roller_motion[2])
            # print(roller_motion_z_dist)
            if (
                not spread
                and roller_motion_z_dist_prev > 0.0001
                and roller_motion_z_dist < 0.0001
            ):
                print("spread!")
                spread = True

            roller_motion_z_dist_prev = roller_motion_z_dist

            roll_angle = 0
            if spread:
                roller_motion_xy_dist = np.linalg.norm(roller_motion[:2])
                if roller_motion_xy_dist > 0:
                    roll_norm = np.cross(
                        roller_motion[:2],
                        (state_shape_seq[i, -1] - state_shape_seq[i, tool_start]),
                    )
                    roll_dir = roll_norm[2] / abs(roll_norm[2])
                    if "large" in args.env:
                        roll_angle = roll_dir * roller_motion_xy_dist / 0.02
                    else:
                        roll_angle = roll_dir * roller_motion_xy_dist / 0.012

            action.extend([roller_motion, [roll_angle, 0, 0]])
        else:
            for j in range(len(args.tool_dim[args.env])):
                tool_dim = args.tool_dim[args.env][j]
                state_diff = (
                    state_shape_seq[i, tool_start] - state_shape_seq[i - 1, tool_start]
                )
                # state_diff_list.append(np.linalg.norm(state_diff))
                action.extend([state_diff, np.zeros(3)])
                tool_start += tool_dim

        actions.append(np.concatenate(action))

    act_seq.append(actions)

    return np.array(act_seq)


def get_normals_from_state(args, state, visualize=False):
    state_normals_list = []

    dough_points = state[: args.n_particles]
    dough_normals = get_normals(dough_points[None])[0]
    state_normals_list.append(dough_normals)

    dough_pcd = o3d.geometry.PointCloud()
    dough_pcd.points = o3d.utility.Vector3dVector(dough_points)
    dough_pcd.normals = o3d.utility.Vector3dVector(dough_normals)

    state_normals_list.append(args.floor_normals)

    floor_pcd = o3d.geometry.PointCloud()
    floor_points = state[args.n_particles : args.n_particles + args.floor_dim]
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.normals = o3d.utility.Vector3dVector(args.floor_normals)

    tool_start = args.n_particles + args.floor_dim
    tool_pcd_list = []
    for k in range(len(args.tool_dim[args.env])):
        tool_dim = args.tool_dim[args.env][k]
        tool_points = state[tool_start : tool_start + tool_dim]
        tool_normals = get_normals(tool_points[None])[0]
        state_normals_list.append(tool_normals)

        tool_pcd = o3d.geometry.PointCloud()
        tool_pcd.points = o3d.utility.Vector3dVector(tool_points)
        tool_pcd.normals = o3d.utility.Vector3dVector(tool_normals)
        tool_pcd_list.append(tool_pcd)

        tool_start += tool_dim

    # import pdb; pdb.set_trace()
    if visualize:
        o3d.visualization.draw_geometries(
            [dough_pcd, floor_pcd, *tool_pcd_list], point_show_normal=True
        )

    return np.concatenate(state_normals_list, axis=0)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True
