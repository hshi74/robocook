import numpy as np
import torch

from datetime import datetime
from utils.data_utils import *
from pytorch3d.transforms import *


def normalize_state(args, state_cur, state_goal, pkg="numpy"):
    if len(state_cur.shape) == 2:
        dim = 0
    elif len(state_cur.shape) == 3:
        dim = 1
    else:
        raise NotImplementedError

    if pkg == "numpy":
        mean_p = np.mean(state_cur, axis=dim)
        std_p = np.std(state_cur, axis=dim)
        state_cur_norm = state_cur - mean_p

        mean_goal = np.mean(state_goal, axis=dim)
        std_goal = np.std(state_goal, axis=dim)
        # if 'alphabet' in args.target_shape_name:
        #     # state_goal_norm = (state_goal - mean_goal) / std_goal
        #     if dim == 0:
        #         std_goal_new = np.concatenate((std_p[:2], std_goal[2:]))
        #     else:
        #         std_goal_new = np.concatenate((std_p[:, :2], std_goal[:, 2:]), axis=1)

        #     state_goal_norm = (state_goal - mean_goal) / std_goal_new
        # else:
        state_goal_norm = state_goal - mean_goal
    else:
        mean_p = torch.mean(state_cur, dim=dim)
        std_p = torch.std(state_cur, dim=dim)
        state_cur_norm = state_cur - mean_p

        mean_goal = torch.mean(state_goal, dim=dim)
        std_goal = torch.std(state_goal, dim=dim)
        # if 'alphabet' in args.target_shape_name:
        #     # state_goal_norm = (state_goal - mean_goal) / std_goal
        #     if dim == 0:
        #         std_goal_new = torch.cat((std_p[:2], std_goal[2:]))
        #     else:
        #         std_goal_new = torch.cat((std_p[:, :2], std_goal[:, 2:]), dim=1)

        #     state_goal_norm = (state_goal - mean_goal) / std_goal_new
        # else:
        state_goal_norm = state_goal - mean_goal

    return state_cur_norm, state_goal_norm


def get_param_bounds(args, tool_params, min_bounds, max_bounds):
    param_bounds = []
    if "gripper" in args.env:
        r = min(max_bounds[:2] - min_bounds[:2]) / 2
        param_bounds.append([-r, r])
    else:
        pos_x_bounds = [-tool_params["x_noise"], tool_params["x_noise"]]
        pos_y_bounds = [-tool_params["y_noise"], tool_params["y_noise"]]
        if "roller" in args.env:
            pos_z_bounds = [
                0.5 * (max_bounds[2] - min_bounds[2]),
                max_bounds[2] - min_bounds[2],
            ]
        else:
            pos_z_bounds = [0.0, max_bounds[2] - min_bounds[2]]

        param_bounds.extend([pos_x_bounds, pos_y_bounds, pos_z_bounds])

    if not "circle" in args.env:
        param_bounds.append(tool_params["rot_range"])

    if "gripper" in args.env:
        param_bounds.append([tool_params["grip_min"], min(r, 0.04)])

    # if 'roller' in args.env:
    #     param_bounds.append(tool_params["roll_range"])

    return torch.FloatTensor(np.array(param_bounds))


def params_to_init_pose(args, center, tool_params, param_seq):
    ee_fingertip_T_mat = torch.FloatTensor(args.ee_fingertip_T_mat)

    init_pose_seq = []
    for params in param_seq:
        if "gripper" in args.env:
            rot_noise = params[1]
            ee_pos = torch.FloatTensor(
                [
                    center[0] - params[0] * torch.sin(rot_noise - np.pi / 4),
                    center[1] + params[0] * torch.cos(rot_noise - np.pi / 4),
                    tool_params["init_h"],
                ]
            )
        else:
            ee_pos = torch.cat(
                (center[:2] + params[:2], torch.FloatTensor([tool_params["init_h"]]))
            )
            if "circle" in args.env:
                rot_noise = torch.zeros(1).squeeze()
            else:
                rot_noise = params[3]

        ee_angle = torch.cat([torch.zeros(2), rot_noise.unsqueeze(0)])
        ee_rot = euler_angles_to_matrix(ee_angle, "XYZ") @ quaternion_to_matrix(
            torch.FloatTensor([0, 1, 0, 0])
        )

        fingertip_mat = ee_fingertip_T_mat[:3, :3] @ ee_rot
        fingermid_pos = (ee_rot @ ee_fingertip_T_mat[:3, 3]) + ee_pos

        fingertip_T_list = []
        for k in range(len(args.tool_dim[args.env])):
            if "gripper" in args.env:
                offset = torch.FloatTensor(
                    [(1 - 2 * k) * tool_params["init_grip"], 0, 0]
                )
                fingertip_pos = (fingertip_mat @ offset) + fingermid_pos
            else:
                fingertip_pos = fingermid_pos.float()

            fingertip_T = torch.cat(
                (
                    torch.cat((fingertip_mat, fingertip_pos.unsqueeze(1)), dim=1),
                    torch.FloatTensor([[0, 0, 0, 1]]),
                ),
                dim=0,
            )
            fingertip_T_list.append(fingertip_T)

        tool_repr = get_tool_repr(args, fingertip_T_list, pkg="torch")
        init_pose_seq.append(tool_repr)

    return torch.stack(init_pose_seq)


def params_to_actions(args, tool_params, param_seq, min_bounds, step=1):
    zero_pad = torch.zeros(3)
    act_seq = []
    for params in param_seq:
        actions = []
        if "gripper" in args.env:
            _, rot_noise, grip_width = params
            grip_rate = (tool_params["init_grip"] - grip_width * 0.5) / (
                tool_params["act_len"] / step
            )
            for _ in range(0, tool_params["act_len"], step):
                x = -grip_rate * torch.sin(rot_noise + np.pi / 4)
                y = grip_rate * torch.cos(rot_noise + np.pi / 4)
                gripper_l_act = torch.cat(
                    [x.unsqueeze(0), y.unsqueeze(0), torch.zeros(1)]
                )
                gripper_r_act = 0 - gripper_l_act
                act = torch.cat((gripper_l_act, zero_pad, gripper_r_act, zero_pad))
                actions.append(act)
        elif "press" in args.env or "punch" in args.env:
            tool_height = 0.01
            press_pos_z = (
                min_bounds[2]
                + params[2]
                + args.tool_center[args.env][0][2]
                + args.ee_fingertip_T_mat[2][3]
                + tool_height
            )
            press_rate = (press_pos_z - tool_params["init_h"]) / (
                tool_params["act_len"] / step
            )
            for _ in range(0, tool_params["act_len"], step):
                press_act = torch.cat([torch.zeros(2), press_rate.unsqueeze(0)])
                act = torch.cat((press_act, zero_pad))
                actions.append(act)
        elif "roller" in args.env:
            if "large" in args.env:
                tool_height = 0.02
            else:
                tool_height = 0.012
            roll_pos_z = (
                min_bounds[2]
                + params[2]
                + args.tool_center[args.env][0][2]
                + args.ee_fingertip_T_mat[2][3]
                + tool_height
            )
            _, _, _, rot_noise = params
            roll_dist = torch.tensor(tool_params["roll_range"])
            press_act_len = tool_params["act_len"] // 2
            press_rate = (roll_pos_z - tool_params["init_h"]) / (press_act_len / step)
            for _ in range(0, press_act_len, step):
                press_act = torch.cat([torch.zeros(2), press_rate.unsqueeze(0)])
                act = torch.cat((press_act, zero_pad))
                actions.append(act)

            roll_z_angle = torch.cat(
                [torch.zeros(2), (rot_noise + np.pi / 4).unsqueeze(0)]
            )
            roll_z_rot_mat = euler_angles_to_matrix(roll_z_angle, "XYZ")
            roll_delta = roll_z_rot_mat @ torch.cat(
                [roll_dist.unsqueeze(0), torch.zeros(2)]
            )

            roll_act_len = tool_params["act_len"] - press_act_len
            for _ in range(0, roll_act_len, step):
                x = roll_delta[0] / (roll_act_len / step)
                y = roll_delta[1] / (roll_act_len / step)
                roll_act = torch.cat([x.unsqueeze(0), y.unsqueeze(0), torch.zeros(1)])
                roll_angle = 0
                roll_act_dist = torch.linalg.norm(roll_act)
                if args.full_repr and roll_act_dist > 0:
                    # roll_norm = torch.cross(roll_act, (torch.FloatTensor([[0, torch.sign(roll_dist), 0]]) @ roll_z_rot_mat).squeeze())
                    roll_dir = torch.sign(roll_dist)
                    if "large" in args.env:
                        roll_angle = roll_dir * roll_act_dist / 0.02
                    else:
                        roll_angle = roll_dir * roll_act_dist / 0.012
                act = torch.cat((roll_act, torch.tensor([roll_angle, 0, 0])))
                actions.append(act)
        else:
            raise NotImplementedError

        act_seq.append(torch.stack(actions))

    return torch.stack(act_seq)


def init_pose_to_params(init_pose_seq):
    # import pdb; pdb.set_trace()
    if not torch.is_tensor(init_pose_seq):
        init_pose_seq = torch.FloatTensor(init_pose_seq)

    mid_point = (init_pose_seq.shape[1] - 1) // 2
    mid_point_seq = (
        init_pose_seq[:, mid_point, :3] + init_pose_seq[:, mid_point, 7:10]
    ) / 2

    rot_seq = torch.atan2(
        init_pose_seq[:, mid_point, 2] - mid_point_seq[:, 2],
        init_pose_seq[:, mid_point, 0] - mid_point_seq[:, 0],
    )

    a = init_pose_seq[:, 0, :3] - init_pose_seq[:, -1, :3]
    b = torch.FloatTensor([[0.0, 1.0, 0.0]]).expand(init_pose_seq.shape[0], -1)
    z_angle_seq = torch.acos(
        (a * b).sum(dim=1)
        / (a.pow(2).sum(dim=1).pow(0.5) * b.pow(2).sum(dim=1).pow(0.5))
    )

    pi = torch.full(rot_seq.shape, np.pi)
    rot_seq = pi - rot_seq
    z_angle_seq = pi - z_angle_seq

    return mid_point_seq, rot_seq, z_angle_seq
