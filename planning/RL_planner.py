import glob
import os
import subprocess
import time
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from control_utils import *
from planner import Planner
from policy import pointnet2_param_cls
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dynamics.gnn import GNN
from dynamics.model import ChamferLoss, EarthMoverLoss
from perception.pcd_utils import *
from planning.sac.agent import Agent
from planning.sac.algorithm import SAC
from planning.sac.network import GaussianPolicy, TwinnedStateActionFunction
from planning.sac.replay_buffer import ReplayBuffer
from planning.sac.utils import RunningMeanStats
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *

from timeit import default_timer as timer


class RLPlanner(Planner):
    def __init__(
        self,
        args,
        tool_params,
        num_steps=2500,  # 100
        batch_size=256,
        memory_size=1000000,
        update_interval=1,
        start_steps=250,
        log_interval=10,
        eval_interval=50,
        seed=0,
        logger=None,
        model_path=None,
    ):
        self.args = args

        self.chamfer_loss = ChamferLoss(args.loss_ord)

        if "sim" in args.planner_type:
            from dynamics.sim import MPM

            self.model = MPM(args)
        else:
            self.model = GNN(args, model_path)

        self.tool_params = tool_params
        self.act_len = int(tool_params["act_len"] / args.time_step)

        self.device = args.device
        self.batch_size = args.control_batch_size

        self.observation_space_shape = (args.n_particles * 3,)
        self.action_space_shape = (args.tool_action_space_size[args.env],)

        self.gamma = 0.99
        self.nstep = 1
        self._replay_buffer = ReplayBuffer(
            memory_size=memory_size,
            state_shape=self.observation_space_shape,
            action_shape=self.action_space_shape,
            gamma=self.gamma,
            nstep=self.nstep,
        )

        self._device = args.device
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval

    def train(self, state_cur, target_shape, max_n_actions):
        self._steps = 0
        self._episodes = 0

        self._best_eval_score = -np.inf

        pbar = tqdm(
            total=float("inf"),
            desc="Training",
            position=0,
            bar_format="{desc}: {n} {bar}",
        )
        while True:
            self._episodes += 1
            episode_return = 0.0
            episode_steps = 0

            pbar.update()

            state = copy.deepcopy(state_cur[0])
            cur_distance = self.evaluate_traj(
                state[None, None, :, :], target_shape, self.args.control_loss_type
            )[0, -1]
            done = False
            while not done:
                if self._start_steps > self._steps:
                    action = torch.zeros(self.param_bounds.shape[0])
                    for i in range(action.shape[0]):
                        action[i] = self.param_bounds[i][0] + (
                            self.param_bounds[i][1] - self.param_bounds[i][0]
                        ) * torch.rand(1)
                    action = action.detach().requires_grad_()
                else:
                    action = self._algo.explore(
                        state.reshape(-1).cpu().detach().numpy(),
                        self.param_bounds.numpy(),
                    )
                    action = torch.from_numpy(action).detach().requires_grad_()

                loss_seqs, state_seqs = self.rollout(
                    action[None, None, :], state, target_shape
                )
                next_state = state_seqs[0, -1]
                reward = cur_distance - loss_seqs[0, -1]
                cur_distance = loss_seqs[0, -1]

                # Set done=True only when the agent fails, ignoring done signal
                # if the agent reach time horizons.
                if episode_steps + 1 >= max_n_actions:
                    masked_done = done = True
                else:
                    masked_done = done

                self._replay_buffer.append(
                    state.reshape(-1).detach().cpu(),
                    action.detach().cpu(),
                    reward.detach().cpu(),
                    next_state.reshape(-1).detach().cpu(),
                    masked_done,
                    episode_done=done,
                )
                self._steps += 1
                episode_steps += 1
                episode_return += reward.detach().cpu()
                state = next_state

                if self._steps >= self._start_steps:
                    # Update online networks.
                    if self._steps % self._update_interval == 0:
                        batch = self._replay_buffer.sample(
                            self._batch_size, self._device
                        )
                        self._algo.update_online_networks(batch, self._writer)

                    # Update target networks.
                    self._algo.update_target_networks()

            # Evaluate.
            if self._episodes % self._eval_interval == 0:
                self.evaluate(state_cur, target_shape, max_n_actions)
                self._algo.save_models(os.path.join(self._model_dir, "final"))

            # We log running mean of training rewards.
            self._train_return.append(episode_return)

            if self._episodes % self._log_interval == 0:
                self._writer.add_scalar(
                    "reward/train", self._train_return.get(), self._steps
                )
                print("train reward", self._train_return.get())

            if self._steps % 300 == 0:
                print(f"current step is {self._steps}")
            if self._steps > self._num_steps:
                break

        self._writer.close()
        print("break")

    def evaluate(self, state_cur, target_shape, max_n_actions):
        state = copy.deepcopy(state_cur[0])
        for _ in range(max_n_actions):
            action = self._algo.exploit(
                state.reshape(-1).cpu().detach().numpy(), self.param_bounds.numpy()
            )
            action = torch.from_numpy(action)
            loss_seqs, state_seqs = self.rollout(
                action[None, None, :], state, target_shape
            )
            next_state = state_seqs[0][-1]
            state = next_state

        episode_return = 0 - loss_seqs[0, -1]
        if episode_return > self._best_eval_score:
            self._best_eval_score = episode_return
            self._algo.save_models(os.path.join(self._model_dir, "best"))
        self._writer.add_scalar("reward/test", episode_return, self._steps)

        print("-" * 60)
        print(f"Num steps: {self._steps} " f"return: {episode_return}")
        print("-" * 60)

    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)

        self._log_dir = rollout_path
        self._model_dir = os.path.join(rollout_path, "model")
        self._summary_dir = os.path.join(rollout_path, "summary")
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self._train_return = RunningMeanStats(self._log_interval)
        self._writer = SummaryWriter(log_dir=self._summary_dir)

        if self.args.debug:
            max_n_actions = 1

        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        max_bounds = torch.max(state_cur, dim=1).values.squeeze().cpu().numpy()
        self.param_bounds = get_param_bounds(
            self.args, self.tool_params, min_bounds, max_bounds
        )

        self._algo = SAC(
            state_dim=self.observation_space_shape[0],
            action_dim=self.action_space_shape[0],
            device=self.device,
            gamma=self.gamma,
            nstep=self.nstep,
            policy_lr=0.0003,
            q_lr=0.0003,
            entropy_lr=0.0003,
            policy_hidden_units=[256, 256],
            q_hidden_units=[256, 256],
            target_update_coef=0.005,
            log_interval=10,
            GaussianPolicy=GaussianPolicy,
            TwinnedStateActionFunction=TwinnedStateActionFunction,
        )

        start = timer()
        with torch.set_grad_enabled(True):
            self.train(state_cur, target_shape, max_n_actions)
        end = timer()
        print(f"Training takes {end - start} seconds")

        pretrained_dict = torch.load(
            os.path.join(self._model_dir, "best", "policy_net.pth"),
            # "/scr/hshi74/projects/robocook/dump/control/control_gripper_sym_rod_robot_v4_surf_nocorr_full/alphabet/K/control_close_max=2_RS_chamfer_Apr-04-23:26:17/000/model/best/policy_net.pth",
            map_location=self.device,
        )
        self._algo._policy_net.load_state_dict(pretrained_dict, strict=False)
        self._algo._policy_net.eval()

        loss_min = np.inf
        param_seq_opt = []
        state = copy.deepcopy(state_cur[0])
        for i in range(max_n_actions):
            action = self._algo.exploit(
                state.reshape(-1).cpu().detach().numpy(), self.param_bounds.numpy()
            )
            action = torch.from_numpy(action)
            loss_seqs, state_seqs = self.rollout(
                action[None, None, :], state, target_shape
            )
            next_state = state_seqs[0][-1]
            loss = loss_seqs[0][-1].detach().cpu()
            param_seq_opt.append(action)
            # print(action, reward)

            if loss < loss_min:
                n_actions_opt = i
                loss_min = loss

            state = next_state

        return torch.stack(param_seq_opt)[: n_actions_opt + 1]

    @profile
    def rollout(self, param_seqs, state_cur, target_shape):
        B = self.batch_size
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().detach().cpu().numpy()
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
