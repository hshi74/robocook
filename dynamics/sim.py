import cv2 as cv
import numpy as np
import open3d as o3d
import os
import subprocess

from datetime import datetime
from perception.pcd_utils import *
from plb.engine.taichi_env import TaichiEnv
from plb.config import load
from sys import platform
from transforms3d.quaternions import axangle2quat
from utils.config import gen_args
from utils.data_utils import load_data
from utils.loss import *
from utils.visualize import *

cfg = load("config/taichi_env/tools.yml")
print(cfg)
env = TaichiEnv(cfg, nn=False, loss=False)
env.initialize()
init_state = env.get_state()
# env.render('plt')


class MPM(object):
    def __init__(self, args):
        # cfg = load(args.sim_config_path)
        # print(cfg)

        self.args = args
        self.n_particles = args.n_particles

        self.action_dim = 0.02
        self.scale = np.array([0.25, 0.2, 0.25])
        self.mid_point = np.array([0.5, 0.1, 0.5])

        # self.env = TaichiEnv(cfg, nn=False, loss=False)
        # self.env.initialize()
        self.env = env
        self.env.set_state(**init_state)
        self.set_parameters(yield_stress=200, E=5e3, nu=0.2)  # 200， 5e3, 0.2
        self.update_camera()

    def set_parameters(self, yield_stress, E, nu):
        self.env.simulator.yield_stress.fill(yield_stress)
        _mu, _lam = E / (2 * (1 + nu)), E * nu / (
            (1 + nu) * (1 - 2 * nu)
        )  # Lame parameters
        self.env.simulator.mu.fill(_mu)
        self.env.simulator.lam.fill(_lam)

    def look_at(self, position):
        target = self.mid_point
        forward = target - position
        forward_proj = [forward[0], 0, forward[2]]
        y_rot = np.arctan2(forward[2], forward[0])
        if forward[0] * forward[2] > 0:
            y_rot += np.pi
        x_rot = np.arccos(
            np.dot(forward, forward_proj)
            / (np.linalg.norm(forward) * np.linalg.norm(forward_proj))
        )
        return x_rot, y_rot

    def update_camera(self):
        camera_pos_1 = (self.mid_point[0] - 0.5, 2.5, self.mid_point[2] - 0.5)
        camera_pos_2 = (self.mid_point[0] + 0.5, 2.5, self.mid_point[2] - 0.5)
        camera_pos_3 = (self.mid_point[0], 0.5, self.mid_point[2] + 2.5)
        camera_pos_4 = (self.mid_point[0], 2.5, self.mid_point[2])

        self.env.render_cfg.defrost()
        self.env.render_cfg.camera_pos_1 = camera_pos_1
        self.env.render_cfg.camera_rot_1 = self.look_at(camera_pos_1)
        self.env.render_cfg.camera_pos_2 = camera_pos_2
        self.env.render_cfg.camera_rot_2 = self.look_at(camera_pos_2)
        self.env.render_cfg.camera_pos_3 = camera_pos_3
        self.env.render_cfg.camera_rot_3 = (0, 0)
        self.env.render_cfg.camera_pos_4 = camera_pos_4
        self.env.render_cfg.camera_rot_4 = (1.57, 0)

    def normalize_state(self, state_cur_dense):
        state_cur = fps(state_cur_dense, n_particles=3000)

        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(state_cur)
        # visualize_o3d([fps_pcd], title='fps_point_cloud')

        self.scale_factor = self.scale[0] / (
            np.max(state_cur[:, 0]) - np.min(state_cur[:, 0])
        )
        self.state_mean = np.mean(state_cur, axis=0)
        state_cur_norm = (state_cur - self.state_mean) * self.scale_factor
        state_cur_norm = np.stack(
            [state_cur_norm[:, 0], state_cur_norm[:, 2], -state_cur_norm[:, 1]]
        ).T
        self.mid_point[1] = np.max(state_cur_norm[:, 1]) - np.min(state_cur_norm[:, 1])
        state_cur_norm = state_cur_norm + self.mid_point

        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(state_cur_norm)
        # visualize_o3d([fps_pcd], title='fps_point_cloud')

        current_state = copy.deepcopy(init_state)
        current_state["state"][0] = state_cur_norm

        return current_state

    def denormalize_state(self, state_pred_norm):
        state_pred = state_pred_norm - self.mid_point
        state_pred = np.stack([state_pred[:, 0], -state_pred[:, 2], state_pred[:, 1]]).T
        state_pred = state_pred / self.scale_factor + self.state_mean

        return state_pred

    @profile
    def rollout(self, state_cur, init_pose_seqs, act_seqs, rollout_path=""):
        # only support gripper and punch at this point
        # TODO: surface sampling should be false to use the simulator
        args = self.args
        state_cur = state_cur.squeeze().cpu().numpy()
        init_pose_seqs = init_pose_seqs.cpu().numpy()
        act_seqs = act_seqs.cpu().numpy()
        B = init_pose_seqs.shape[0]
        current_state = self.normalize_state(state_cur)
        state_seq_list = []
        for b in range(B):
            self.env.set_state(**current_state)
            state_pred_list = []
            for i in range(act_seqs.shape[1]):
                if "gripper" in args.env:
                    tool_center_list = []
                    tool_start = 0
                    for j in range(len(args.tool_dim[args.env])):
                        tool_dim = args.tool_dim[args.env][j]
                        tool_center_list.append(
                            np.mean(
                                init_pose_seqs[
                                    b, i, tool_start : tool_start + tool_dim
                                ],
                                axis=0,
                            )
                        )
                        tool_start += tool_dim
                    tool_center = np.stack(tool_center_list)

                    tool_vec = tool_center[1] - tool_center[0]
                    tool_rot_theta = np.arctan2(tool_vec[1], tool_vec[0]) - np.pi / 2
                    tool_rot = axangle2quat([0, 1, 0], tool_rot_theta)
                else:
                    tool_center = np.mean(init_pose_seqs[b, i], axis=0)[None]
                    if "square" in args.env:
                        plane_vec = tool_center[0] - init_pose_seqs[b, i, 0]
                        tool_rot_theta = (
                            np.arctan2(plane_vec[1], plane_vec[0]) - np.pi / 4
                        )
                        tool_rot = axangle2quat([0, 1, 0], tool_rot_theta)
                    else:
                        tool_rot = [1.0, 0.0, 0.0, 0.0]

                tool_center_norm = (
                    tool_center - np.mean(state_cur, axis=0)
                ) * self.scale_factor
                tool_center_norm = np.stack(
                    [
                        tool_center_norm[:, 0],
                        tool_center_norm[:, 2],
                        -tool_center_norm[:, 1],
                    ]
                ).T
                tool_center_norm = tool_center_norm + self.mid_point

                for j in range(len(args.tool_sim_primitive_mapping[args.env])):
                    prim_idx = args.tool_sim_primitive_mapping[args.env][j]
                    self.env.primitives.primitives[prim_idx].set_state(
                        0, [*tool_center_norm[j], *tool_rot]
                    )

                for j in range(act_seqs.shape[2]):
                    action_list = []
                    for k in range(len(args.tool_dim[args.env])):
                        action_list.append(
                            np.array(
                                [
                                    act_seqs[b, i, j, 6 * k],
                                    act_seqs[b, i, j, 6 * k + 2],
                                    -act_seqs[b, i, j, 6 * k + 1],
                                    0.0,
                                    0.0,
                                    0.0,
                                ]
                            )
                            * self.scale_factor
                            / self.action_dim
                        )
                    if "gripper" in args.env:
                        prim_idx_1, prim_idx_2 = args.tool_sim_primitive_mapping[
                            args.env
                        ]
                        action = np.concatenate(
                            [
                                np.zeros(6 * prim_idx_1),
                                action_list[0],
                                np.zeros(6 * (prim_idx_2 - prim_idx_1 - 1)),
                                action_list[1],
                                np.zeros(6 * (6 - prim_idx_2 - 1)),
                            ],
                            axis=0,
                        )
                    else:
                        prim_idx = args.tool_sim_primitive_mapping[args.env][0]
                        action = np.concatenate(
                            [
                                np.zeros(6 * prim_idx),
                                action_list[0],
                                np.zeros(6 * (6 - prim_idx - 1)),
                            ],
                            axis=0,
                        )

                    # print(self.env.primitives.primitives, action)
                    self.env.step(action)

                    if len(rollout_path) > 0:
                        img = self.env.render_multi(mode="rgb_array", spp=3)
                        rgb, depth = img[0], img[1]
                        idx = act_seqs.shape[2] * i + j

                        for cam_idx in range(4):
                            cv.imwrite(
                                f"{rollout_path}/{idx:03d}_rgb_{cam_idx+1}.png",
                                rgb[cam_idx][..., ::-1],
                            )

                    x = self.env.simulator.get_x(0)
                    # step_size = len(x) // self.n_particles
                    # state_pred_norm = np.array(x[::step_size])
                    # dough_pcd = o3d.geometry.PointCloud()
                    # dough_pcd.points = o3d.utility.Vector3dVector(x)

                    # if j == act_seqs.shape[2] - 1:
                    #     visualize = False
                    # else:
                    #     visualize = False

                    # surf_mesh = alpha_shape_mesh_reconstruct(dough_pcd, alpha=0.02, visualize=False)
                    # surf_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(surf_mesh, self.n_particles)
                    # visualize_o3d([surf_mesh], title='surf_pcd')

                    # state_pred_norm = np.asarray(surf_pcd.points)
                    state_pred_norm = x
                    state_pred = self.denormalize_state(state_pred_norm)
                    state_pred_list.append(state_pred)

                    # if visualize:
                    #     state_pred_pcd = o3d.geometry.PointCloud()
                    #     state_pred_pcd.points = o3d.utility.Vector3dVector(state_pred)
                    #     visualize_o3d([state_pred_pcd])

            state_seq = np.stack(state_pred_list)[:: args.time_step]
            state_seq_list.append(state_seq)

            if len(rollout_path) > 0:
                for cam_idx in range(4):
                    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            f"{rollout_path}/%03d_rgb_{cam_idx+1}.png",
                            "-c:v",
                            "libx264",
                            "-vf",
                            "fps=15",
                            "-pix_fmt",
                            "yuv420p",
                            f"{rollout_path}/{args.env}_sim_vid_{cam_idx+1}_{time_now}.mp4",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                # os.system(f'rm {rollout_path}/*.png')
                subprocess.run(f"rm {rollout_path}/*.png", shell=True)

        states_pred_array = torch.tensor(
            np.stack(state_seq_list), dtype=torch.float32, device=args.device
        )
        return states_pred_array, None, None


if __name__ == "__main__":
    import yaml
    from planning.control_utils import *
    from datetime import datetime

    args = gen_args()

    with open(f"misc/{args.env}_param_seq.yml", "r") as f:
        param_seq_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open("config/tool_plan_params.yml", "r") as f:
        tool_params_dict = yaml.load(f, Loader=yaml.FullLoader)

    param_seq = torch.tensor(param_seq_dict[args.env], dtype=torch.float32)
    tool_params = tool_params_dict[args.env]

    state_cur = load_data(args.data_names, "target_shapes/alphabet_sim/start/start.h5")[
        0
    ][: args.n_particles]

    min_bounds = np.min(state_cur, axis=0)
    init_pose_seq = params_to_init_pose(
        args, np.mean(state_cur, axis=0), tool_params, param_seq
    ).numpy()
    act_seq = params_to_actions(
        args, tool_params, param_seq, min_bounds, step=1
    ).numpy()

    rollout_path = (
        f'misc/gripper_sym_rod/test_{datetime.now().strftime("%b-%d-%H:%M:%S")}'
    )
    os.system(f"mkdir -p {rollout_path}")

    sim = MPM(args)
    # state_cur: [N, state_dim]
    # init_pose_seqs: [B, n_grip, n_shape, 14]
    # act_seqs: [B, n_grip, n_steps, 12]
    state_seqs = sim.rollout(
        state_cur, init_pose_seq[None], act_seq[None], rollout_path=rollout_path
    )
