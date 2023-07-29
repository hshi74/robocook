import copy
import itertools
import numpy as np
import os
import pickle
import skopt
import subprocess
import sys
import torch
import torch.nn.functional as F
import yaml

torch.set_printoptions(sci_mode=False)

from control_utils import *
from datetime import datetime
from dynamics.model import ChamferLoss, EarthMoverLoss, HausdorffLoss
from dynamics.gnn import GNN
from perception.pcd_utils import *
from planner import Planner
from tqdm import tqdm
from utils.data_utils import *

# from utils.Density_aware_Chamfer_Distance.utils_v2.model_utils import calc_dcd
from utils.loss import *
from utils.visualize import *


class ModelBasedPlanner(Planner):
    def __init__(self, args, tool_params, model_path=None):
        self.args = args

        self.chamfer_loss = ChamferLoss(args.loss_ord)
        self.emd_loss = EarthMoverLoss(args.loss_ord)
        self.h_loss = HausdorffLoss(args.loss_ord)

        if "sim" in args.planner_type:
            from dynamics.sim import MPM

            self.model = MPM(args)
        else:
            self.model = GNN(args, model_path)
        self.tool_params = tool_params
        self.act_len = int(tool_params["act_len"] / args.time_step)

        self.device = args.device
        self.batch_size = args.control_batch_size
        # self.RS_sample_size = args.RS_sample_size
        self.RS_elite_size_per_act = args.RS_elite_size_per_act
        self.CEM_sample_size = args.CEM_sample_size
        self.CEM_elite_size = args.CEM_elite_size
        self.CEM_sample_iter = args.CEM_sample_iter
        self.CEM_decay_factor = args.CEM_decay_factor

        if "gripper_asym" in args.env:
            self.CEM_param_var = [1e-1, 1e-1, 1e-2]
            self.sample_ratio = {1: (4, 8, 4)}
        elif "gripper_sym" in args.env:
            self.CEM_param_var = [1e-1, 1e-1, 1e-2]
            self.sample_ratio = {1: (4, 8, 4)}
        elif "press" in args.env or "punch" in args.env:
            if "circle" in args.env:
                self.CEM_param_var = [1e-2, 1e-2, 1e-2]
                self.sample_ratio = {1: (4, 8, 4)}
            else:
                self.CEM_param_var = [1e-2, 1e-2, 1e-2, 1e-1]
                self.sample_ratio = {1: (4, 4, 4, 2)}
        elif "roller" in args.env:
            self.CEM_param_var = [1e-2, 1e-2, 1e-2, 1e-1]
            self.sample_ratio = {1: (4, 4, 2, 4)}
        else:
            raise NotImplementedError

        if args.debug:
            # self.RS_sample_size = 8
            self.RS_elite_size_per_act = 1
            self.CEM_sample_size = 8
            self.CEM_elite_size = 2
            self.CEM_sample_iter = 2
            self.CEM_decay_factor = 0.5
            if "roller" in args.env:
                self.sample_ratio = {1: (1, 1, 1, 4, 2)}
            elif "square" in args.env:
                self.sample_ratio = {1: (1, 1, 4, 2)}
            else:
                self.sample_ratio = {1: (2, 2, 2)}

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)

        if self.args.debug:
            max_n_actions = 1

        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        max_bounds = torch.max(state_cur, dim=1).values.squeeze().cpu().numpy()
        self.param_bounds = get_param_bounds(
            self.args, self.tool_params, min_bounds, max_bounds
        )

        param_seq_pool, loss_seq_pool, state_seq_pool = self.random_search(
            state_cur, target_shape, max_n_actions
        )

        if loss_seq_pool[0] > rs_loss_threshold:
            return param_seq_pool[0]

        if self.args.optim_algo == "RS":
            sort_ind = torch.argsort(loss_seq_pool)
            # print(f"Selected idx: {sort_ind[0]} with loss {loss_seq_pool[sort_ind[0]]}")
            param_seq_opt = param_seq_pool[sort_ind[0]]

        elif self.args.optim_algo == "CEM":
            param_seq_CEM, loss_seq_CEM = self.optimize_CEM(
                (param_seq_pool, loss_seq_pool, state_seq_pool), state_cur, target_shape
            )
            sort_ind = torch.argsort(loss_seq_CEM)
            # print(f"Selected idx: {sort_ind[0]} with loss {loss_seq_CEM[sort_ind[0]]}")
            param_seq_opt = param_seq_CEM[sort_ind[0]]

        elif self.args.optim_algo == "GD":
            with torch.set_grad_enabled(True):
                param_seq_opt = self.optimize_GD(
                    param_seq_pool, state_cur, target_shape
                )

        elif self.args.optim_algo == "CEM_BEFORE_GD":
            (
                param_seq_CEM,
                loss_seq_CEM,
            ) = self.optimize_CEM(
                (param_seq_pool, loss_seq_pool, state_seq_pool), state_cur, target_shape
            )
            with torch.set_grad_enabled(True):
                param_seq_opt = self.optimize_GD(param_seq_CEM, state_cur, target_shape)

        elif self.args.optim_algo == "CEM_GUIDED_GD":
            (
                param_seq_CEM,
                loss_seq_CEM,
            ) = self.optimize_CEM(
                (param_seq_pool, loss_seq_pool, state_seq_pool), state_cur, target_shape
            )
            with torch.set_grad_enabled(True):
                param_seq_opt = self.optimize_GD(
                    param_seq_pool, state_cur, target_shape, guide=param_seq_CEM
                )

        else:
            raise NotImplementedError

        return param_seq_opt

    @profile
    def rollout(self, param_seqs, state_cur, target_shape):
        B = self.batch_size
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        init_pose_seqs = []
        act_seqs = []
        for param_seq in param_seqs:
            init_pose_seq = params_to_init_pose(
                self.args, self.center, self.tool_params, param_seq
            )
            act_seq = params_to_actions(
                self.args, self.tool_params, param_seq, min_bounds
            )
            init_pose_seqs.append(init_pose_seq)
            act_seqs.append(act_seq)

        init_pose_seqs = torch.stack(init_pose_seqs)
        act_seqs = torch.stack(act_seqs)

        n_batch = int(np.ceil(param_seqs.shape[0] / B))
        state_seqs = []
        loss_seqs = []
        batch_iterator = tqdm(range(n_batch)) if n_batch > 2 else range(n_batch)
        for i in batch_iterator:
            start = B * i
            end = min(B * (i + 1), param_seqs.shape[0])
            state_seq, _, _ = self.model.rollout(
                state_cur, init_pose_seqs[start:end], act_seqs[start:end]
            )
            loss_seq = self.evaluate_traj(
                state_seq, target_shape, self.args.control_loss_type
            )
            state_seqs.append(state_seq)
            loss_seqs.append(loss_seq)

        return torch.cat(loss_seqs, dim=0), torch.cat(state_seqs, dim=0)

    @profile
    def evaluate_traj(
        self,
        state_seqs,  # [n_sample, n_steps, n_particles, state_dim]
        target_shape,
        loss_type,
    ):
        loss_seqs = []
        for i in range(state_seqs.shape[0]):
            n_actions = int(np.ceil(state_seqs.shape[1] / self.act_len))
            loss_seq = []
            for j in range(n_actions):
                state_idx = min(state_seqs.shape[1] - 1, (j + 1) * self.act_len - 1)
                if "sim" in self.args.planner_type:
                    dense_pcd = o3d.geometry.PointCloud()
                    dense_pcd.points = o3d.utility.Vector3dVector(
                        state_seqs[i, state_idx].cpu().numpy()
                    )

                    surf_mesh = alpha_shape_mesh_reconstruct(
                        dense_pcd, alpha=0.02, visualize=False
                    )
                    surf_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                        surf_mesh, self.args.n_particles
                    )
                    # visualize_o3d([surf_mesh], title='surf_pcd')

                    state_cur = torch.tensor(
                        np.asarray(surf_pcd.points),
                        device=self.args.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                else:
                    state_cur = state_seqs[i, state_idx].unsqueeze(0)

                if "surf" in self.args.tool_type:
                    state_goal = torch.tensor(
                        target_shape["surf"],
                        device=self.args.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                else:
                    state_goal = torch.tensor(
                        target_shape["sparse"],
                        device=self.args.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)

                state_cur_norm, state_goal_norm = normalize_state(
                    self.args, state_cur, state_goal, pkg="torch"
                )

                if loss_type == "emd":
                    loss = self.emd_loss(state_cur_norm, state_goal_norm).cpu()
                elif loss_type == "chamfer":
                    loss = self.chamfer_loss(state_cur_norm, state_goal_norm).cpu()
                elif loss_type == "chamfer_emd":
                    loss = 0
                    loss += (
                        self.args.emd_weight
                        * self.emd_loss(state_cur_norm, state_goal_norm).cpu()
                    )
                    loss += (
                        self.args.chamfer_weight
                        * self.chamfer_loss(state_cur_norm, state_goal_norm).cpu()
                    )
                elif loss_type == "L1_pos":
                    loss = F.l1_loss(state_cur_norm, state_goal_norm).cpu()
                elif loss_type == "iou":  # complement of IOU
                    state_cur_upsample = upsample(state_cur[0], visualize=False)
                    loss = 1 - iou(
                        state_cur_upsample,
                        target_shape["dense"],
                        voxel_size=0.003,
                        visualize=False,
                    )
                    loss = torch.tensor([loss], dtype=torch.float32)
                elif loss_type == "soft_iou":
                    loss = 1 - soft_iou(
                        state_cur_norm, state_goal_norm, size=8, pkg="torch", soft=True
                    )
                else:
                    raise NotImplementedError

                loss_seq.append(loss)

            loss_seqs.append(torch.stack(loss_seq))

        loss_seqs = torch.stack(loss_seqs).cpu()

        return loss_seqs

    def eval_soln(self, param_seq, state_cur, target_shape):
        # evaluate and store results
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        init_pose_seq = params_to_init_pose(
            self.args, self.center, self.tool_params, param_seq
        )
        act_seq = params_to_actions(self.args, self.tool_params, param_seq, min_bounds)

        if "sim" in self.args.planner_type:
            state_seq, _, _ = self.model.rollout(
                state_cur,
                init_pose_seq.unsqueeze(0),
                act_seq.unsqueeze(0),
                os.path.join(self.rollout_path, "anim"),
            )
        else:
            state_seq, _, _ = self.model.rollout(
                state_cur, init_pose_seq.unsqueeze(0), act_seq.unsqueeze(0)
            )
        # attn_mask_pred = np.squeeze(attn_mask_pred.cpu().numpy()[0])
        # attn_mask_pred = np.concatenate((np.zeros((self.args.n_his, self.args.n_particles)), attn_mask_pred), axis=0)

        # rels_pred = rels_pred[0]
        # max_n_rel = max([rels.shape[0] for rels in rels_pred])
        # for i in range(len(rels_pred)):
        #     rels_pred[i] = np.concatenate((np.zeros((max_n_rel - rels_pred[i].shape[0],
        #         rels_pred[i].shape[1]), dtype=np.int16), rels_pred[i]), axis=0)

        # rels_pred = np.stack(rels_pred)
        # rels_pred = np.concatenate((rels_pred, np.zeros((self.args.n_his, max_n_rel, rels_pred[0].shape[1]),
        #     dtype=np.int16)), axis=0)

        self.render_state(state_seq.squeeze()[-1:], target_shape)
        loss_gnn = self.evaluate_traj(
            state_seq, target_shape, self.args.control_loss_type
        )[0][-1].item()
        # chamfer_loss = self.evaluate_traj(state_seq, target_shape, 'chamfer')[0][-1].item()
        # emd_loss = self.evaluate_traj(state_seq, target_shape, 'emd')[0][-1].item()
        # iou_loss = self.evaluate_traj(state_seq, target_shape, 'iou')[0][-1].item()

        title = f"{self.args.env.upper()}: "
        for i in range(0, len(param_seq), 2):
            if i > 0:
                title += "\n"
            title += f"{[[round(x.item(), 3) for x in y] for y in param_seq[i:i+2]]}"
        title += f" -> {round(loss_gnn, 4)}"
        act_seq_sparse = params_to_actions(
            self.args, self.tool_params, param_seq, min_bounds, step=self.args.time_step
        )
        state_seq_wshape = add_shape_to_seq(
            self.args,
            state_seq.squeeze().cpu().numpy(),
            init_pose_seq.cpu().numpy(),
            act_seq_sparse.cpu().numpy(),
        )

        pkl_path = os.path.join(
            self.rollout_path,
            "anim_args",
            f'{self.args.env}_anim_{datetime.now().strftime("%b-%d-%H:%M:%S")}_args.pkl',
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "args": self.args,
                    "row_titles": [title],
                    "state_seqs": [state_seq_wshape],
                    "target": target_shape["surf"],
                },
                f,
            )
        p = subprocess.Popen(
            ["python", "utils/visualize.py", pkl_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # render_anim(self.args, [title], [state_seq_wshape], target=target_shape['sparse'],
        #     attn_mask_pred=[attn_mask_pred], rels_pred=[rels_pred],
        #     path=os.path.join(self.rollout_path, 'anim', f'{anim_name}.mp4'))

        with open(
            os.path.join(
                self.rollout_path, "param_seqs", f"{self.args.env}_param_seq.yml"
            ),
            "w",
        ) as f:
            yaml.dump({self.args.env: param_seq.tolist()}, f, default_flow_style=True)

        info_dict = {
            "loss": [loss_gnn],
            "subprocess": [p],
        }

        return state_seq_wshape, info_dict

    @profile
    def random_search(self, state_cur, target_shape, max_n_actions):
        def grid_sample(n_actions):
            param_sample_list = []
            for i in range(self.param_bounds.shape[0]):
                param_samples = np.linspace(
                    *self.param_bounds[i],
                    num=self.sample_ratio[n_actions][i] + 1,
                    endpoint=False,
                )[1:]
                param_sample_list.append(param_samples)

            param_grid = np.meshgrid(*param_sample_list)
            param_samples = np.vstack([x.ravel() for x in param_grid]).T

            if n_actions == 1:
                return np.expand_dims(param_samples, axis=1)
            else:
                return np.array(
                    list(itertools.product(param_samples, repeat=n_actions))
                )

        param_seqs = None
        for _ in range(max_n_actions):
            param_seq_sample = grid_sample(1)
            if param_seqs is None:
                param_seqs = param_seq_sample
            else:
                np.random.shuffle(param_seq_sample)
                param_seqs = np.concatenate((param_seqs, param_seq_sample), axis=1)

        # param_seqs: numpy before and torch after
        param_seqs = torch.tensor(param_seqs, dtype=torch.float32)
        loss_seq_list, state_seq_list = self.rollout(
            param_seqs, state_cur, target_shape
        )
        best_loss_ind = torch.argsort(loss_seq_list, dim=0)[
            : self.RS_elite_size_per_act
        ]
        loss_opt = []
        param_seq_opt = []
        state_seq_opt = []
        for i in range(max_n_actions):
            loss_opt.extend([loss_seq_list[x][i] for x in best_loss_ind[:, i]])
            param_seq_opt.extend([param_seqs[x][: i + 1] for x in best_loss_ind[:, i]])
            state_seq_opt.extend(
                [
                    state_seq_list[x][: (i + 1) * self.act_len]
                    for x in best_loss_ind[:, i]
                ]
            )

        loss_opt = torch.stack(loss_opt)
        sort_ind = torch.argsort(loss_opt)
        loss_opt = loss_opt[sort_ind[:max_n_actions]]
        param_seq_opt = [param_seq_opt[x] for x in sort_ind[:max_n_actions]]
        state_seq_opt = [state_seq_opt[x] for x in sort_ind[:max_n_actions]]

        self.render_selections(
            "RS", state_cur, param_seq_opt, loss_opt, state_seq_opt, target_shape
        )

        return param_seq_opt, loss_opt, state_seq_opt

    @profile
    def optimize_CEM(
        self,
        soln_pool,
        state_cur,
        target_shape,
        first_iter_threshold=5e-4,
        plateau_threshold=3e-4,
        plateau_max_iter=5,
    ):
        # https://github.com/martius-lab/iCEM

        param_seq_opt, loss_opt, state_seq_opt = [], [], []
        traj_loss_dict = {}
        for i, (param_seq_cur, loss_cur, state_seq_cur) in enumerate(zip(*soln_pool)):
            n_actions = param_seq_cur.shape[0]
            best_loss, best_param_seq, best_state_seq = (
                loss_cur,
                param_seq_cur,
                state_seq_cur,
            )

            plateau_iter, traj_loss_list = 0, []
            param_seq_mean = param_seq_cur.numpy().flatten()
            param_seq_var = np.tile(self.CEM_param_var, n_actions)
            sample_size = self.CEM_sample_size * n_actions
            interrupt = ""
            for j in range(self.CEM_sample_iter):
                param_seq_samples = np.random.multivariate_normal(
                    mean=param_seq_mean, cov=np.diag(param_seq_var), size=sample_size
                )

                param_seq_CEM = torch.tensor(
                    param_seq_samples, dtype=torch.float32
                ).view(sample_size, n_actions, -1)
                param_seq_CEM = torch.clamp(
                    param_seq_CEM,
                    min=self.param_bounds[:, 0],
                    max=self.param_bounds[:, 1],
                )
                loss_list, state_seq_list = self.rollout(
                    param_seq_CEM, state_cur, target_shape
                )

                loss_list = loss_list[:, -1]

                sort_ind = torch.argsort(loss_list)
                traj_loss_list.append(loss_list[sort_ind[0]])

                param_seq_CEM = param_seq_CEM.reshape(sample_size, -1)
                param_seq_CEM_elite = param_seq_CEM[
                    sort_ind[: self.CEM_elite_size]
                ].numpy()
                param_seq_mean = np.mean(param_seq_CEM_elite, axis=0)
                param_seq_var = np.var(param_seq_CEM_elite, axis=0)

                sample_size = max(
                    2 * self.CEM_elite_size, int(self.CEM_decay_factor * sample_size)
                )

                # check the improvement of absolute scale
                if (
                    j == 0
                    and i != 0
                    and best_loss - loss_list[sort_ind[0]] < first_iter_threshold
                ):
                    interrupt = f"---> Stop criterion 1: not the best RS Traj and < {first_iter_threshold} improvement in the first iteration!"

                if best_loss - loss_list[sort_ind[0]] < plateau_threshold:
                    plateau_iter += 1
                else:
                    plateau_iter = 0

                # break after not improving for 3 iterations
                if len(loss_opt) > 0 and best_loss > min(loss_opt):
                    if plateau_iter >= 0.5 * plateau_max_iter:
                        interrupt = (
                            f"---> Stop criterion 2.a: not currently the best traj and improving "
                            + f"less than {plateau_threshold} for {int(np.ceil(0.5 * plateau_max_iter))} iterations!"
                        )
                else:
                    if plateau_iter >= plateau_max_iter:
                        interrupt = (
                            f"---> Stop criterion 2.b: currently the best traj and improving "
                            + f"less than {plateau_threshold} for {plateau_max_iter} iterations!"
                        )

                if loss_list[sort_ind[0]] < best_loss:
                    best_loss = loss_list[sort_ind[0]]
                    best_state_seq = state_seq_list[sort_ind[0]]
                    best_param_seq = param_seq_CEM[sort_ind[0]]

                if len(interrupt) > 0:
                    break

            if len(interrupt) > 0:
                print(interrupt)

            print(
                f"From: {[round(x.item(), 3) for x in param_seq_cur.flatten()]} -> {round(loss_cur.item(), 4)}\n"
                + f"To: {[round(x.item(), 3) for x in best_param_seq.flatten()]} -> {round(best_loss.item(), 4)}"
            )

            param_seq_opt.append(best_param_seq.reshape(n_actions, -1))
            loss_opt.append(best_loss)
            state_seq_opt.append(best_state_seq)

            traj_loss_dict[f"Traj {i}"] = traj_loss_list

        loss_opt = torch.tensor(loss_opt, dtype=torch.float32)

        plot_eval_loss(
            "Control CEM Loss",
            traj_loss_dict,
            path=os.path.join(
                self.rollout_path, "optim_plots", f"{self.args.env}_CEM_loss"
            ),
        )

        self.render_selections(
            "CEM", state_cur, param_seq_opt, loss_opt, state_seq_opt, target_shape
        )

        return param_seq_opt, loss_opt

    def render_selections(
        self, name, state_cur, param_seq_opt, loss_opt, state_seq_opt, target_shape
    ):
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        row_titles = []
        state_seq_wshape_list = []
        for i in range(len(param_seq_opt)):
            title = f"{name} {i}: "
            for j in range(0, param_seq_opt[i].shape[0], 2):
                if j > 0:
                    title += "\n"
                title += f"{[[round(x.item(), 3) for x in y] for y in param_seq_opt[i][j:j+2]]}"
            title += f" -> {round(loss_opt[i].item(), 4)}"
            row_titles.append(title)
            init_pose_seq = params_to_init_pose(
                self.args, self.center, self.tool_params, param_seq_opt[i]
            )
            act_seq = params_to_actions(
                self.args,
                self.tool_params,
                param_seq_opt[i],
                min_bounds,
                step=self.args.time_step,
            )
            state_seq_wshape = add_shape_to_seq(
                self.args,
                state_seq_opt[i].cpu().numpy(),
                init_pose_seq.cpu().numpy(),
                act_seq.cpu().numpy(),
            )
            state_seq_wshape_list.append(state_seq_wshape)

        print(f"{name} best loss seqs: {loss_opt}")
        print(f"{name} best param seqs: {param_seq_opt}")

        pkl_path = os.path.join(
            self.rollout_path,
            "anim_args",
            f'{self.args.env}_anim_{name}_{datetime.now().strftime("%b-%d-%H:%M:%S")}_args.pkl',
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "args": self.args,
                    "row_titles": row_titles,
                    "state_seqs": state_seq_wshape_list,
                    "target": target_shape["surf"],
                },
                f,
            )
        subprocess.Popen(
            ["python", "utils/visualize.py", pkl_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # render_anim(self.args, row_titles, state_seq_wshape_list, target=target_shape['sparse'],
        #     path=os.path.join(self.rollout_path, 'anim', f'{self.args.env}_anim_{name}.mp4'))

    def render_state(
        self, state_cur, target_shape, state_pred=None, pred_err=0, frame_suffix="model"
    ):
        if "surf" in self.args.tool_type:
            state_goal = torch.tensor(
                target_shape["surf"], device=self.args.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            state_goal = torch.tensor(
                target_shape["sparse"], device=self.args.device, dtype=torch.float32
            ).unsqueeze(0)

        state_cur_norm, state_goal_norm = normalize_state(
            self.args, state_cur, state_goal, pkg="torch"
        )

        if frame_suffix == "before" and state_pred is not None:
            state_pred_norm = (state_pred - torch.mean(state_pred, dim=1)) / torch.std(
                state_pred, dim=1
            )
            render_frames(
                self.args,
                [
                    f"State",
                    f"State Pred={round(pred_err.item(), 6)}",
                    "State Normalized",
                    "State Pred Normalized",
                ],
                [
                    state_cur.cpu()[-1:],
                    state_pred.cpu()[-1:],
                    state_cur_norm.cpu()[-1:],
                    state_pred_norm.cpu()[-1:],
                ],
                axis_off=False,
                focus=[True, True, True, True],
                target=[
                    state_goal.cpu()[0],
                    state_goal.cpu()[0],
                    state_goal_norm.cpu()[0],
                    state_goal_norm.cpu()[0],
                ],
                path=os.path.join(self.rollout_path, "states"),
                name=f"{self.args.env}_state_{frame_suffix}.png",
            )
        else:
            render_frames(
                self.args,
                [f"State", "State Normalized"],
                [state_cur.cpu()[-1:], state_cur_norm.cpu()[-1:]],
                axis_off=False,
                focus=[True, True],
                target=[state_goal.cpu()[0], state_goal_norm.cpu()[0]],
                path=os.path.join(self.rollout_path, "states"),
                name=f"{self.args.env}_state_{frame_suffix}.png",
            )

    def optimize_GD(self, param_seq_pool, state_cur, target_shape, guide=None):
        from torchmin import minimize

        param_seq_opt = []
        loss_list_dict = {}
        loss_opt_min = float("inf")
        best_idx = 0
        for i in range(len(param_seq_pool)):
            print(f"Trajectory {i+1}/{len(param_seq_pool)}:")

            param_seq_cand = param_seq_pool[i].detach().requires_grad_()
            loss_list = []

            def loss_func(param_seq):
                nonlocal loss_list
                # print(f"Params: {param_seq_opt}")
                # param_seq_clamped = torch.clamp(param_seq, min=self.param_bounds[:, 0], max=self.param_bounds[:, 1])
                loss, state_seq = self.rollout(
                    param_seq.unsqueeze(0), state_cur, target_shape
                )
                if guide is not None:
                    loss += 0.2 * torch.linalg.norm(param_seq - guide[i])
                loss_list.append(loss[:, -1].item())
                return loss[:, -1]

            res = minimize(
                loss_func,
                param_seq_cand.contiguous(),
                method="l-bfgs",
                options=dict(lr=1e-1, line_search="strong-wolfe"),
                disp=2,
            )

            # print(res)
            # param_seq_clamped = torch.clamp(
            #     res.x, min=self.param_bounds[:, 0], max=self.param_bounds[:, 1])
            param_seq_opt.append(res.x)
            loss_opt = res.fun

            loss_list_dict[f"Traj {i}"] = loss_list

            if loss_opt.item() < loss_opt_min:
                loss_opt_min = loss_opt.item()
                best_idx = i

        plot_eval_loss(
            "Control GD Loss",
            loss_list_dict,
            path=os.path.join(
                self.rollout_path, "optim_plots", f"{self.args.env}_GD_loss"
            ),
        )

        return param_seq_opt[best_idx]
