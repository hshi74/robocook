import glob
import os
import numpy as np
import subprocess
import torch
import yaml
import pdb

from control_utils import *
from dynamics.gnn import GNN
from dynamics.model import ChamferLoss, EarthMoverLoss
from planner import Planner
from policy import pointnet2_param_cls
from tqdm import tqdm
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


class LearnedPlanner(Planner):
    def __init__(self, args, name, tool_model_path_list, tool_params):
        self.chamfer_loss = ChamferLoss(args.loss_ord)
        self.emd_loss = EarthMoverLoss(args.loss_ord)

        self.device = args.device
        self.optim_algo = args.optim_algo
        self.batch_size = args.control_batch_size
        self.CEM_sample_size = args.CEM_sample_size
        self.CEM_elite_size = args.CEM_elite_size
        self.CEM_sample_iter = args.CEM_sample_iter
        self.CEM_decay_factor = args.CEM_decay_factor

        if "gripper" in name:
            self.CEM_param_var = [1e-4, 1e-3, 1e-4]
        elif "press" in name or "punch" in name:
            if "circle" in name:
                self.CEM_param_var = [1e-4, 1e-4, 1e-4]
            else:
                self.CEM_param_var = [1e-4, 1e-4, 1e-4, 1e-3]
        elif "roller" in name:
            self.CEM_param_var = [1e-4, 1e-4, 1e-4, 1e-3]
        else:
            raise NotImplementedError

        self.name = name
        self.tool_params = tool_params

        self.planner_dict = {"args": [], "model": []}
        for tool_model_path in tool_model_path_list:
            tool_args = copy.deepcopy(args)
            tool_args_dict = np.load(
                f"{tool_model_path}_args.npy", allow_pickle=True
            ).item()
            tool_args.__dict__ = tool_args_dict
            tool_args.env = name
            self.planner_dict["args"].append(tool_args)

            planner = pointnet2_param_cls.get_model(tool_args)
            planner.apply(inplace_relu)
            planner = planner.to(self.device)
            pretrained_dict = torch.load(
                f"{tool_model_path}.pth", map_location=self.device
            )
            planner.load_state_dict(pretrained_dict, strict=False)
            planner.eval()
            self.planner_dict["model"].append(planner)

        dy_args_dict = np.load(
            f"{tool_args.plan_dataf}/args.npy", allow_pickle=True
        ).item()
        self.dy_args = copy.deepcopy(tool_args)
        self.dy_args.__dict__ = dy_args_dict
        self.dy_args.ee_fingertip_T_mat = np.array(
            [
                [0.707, 0.707, 0, 0],
                [-0.707, 0.707, 0, 0],
                [0, 0, 1, 0.1034],
                [0, 0, 0, 1],
            ]
        )
        self.dy_args.env = name
        self.gnn = GNN(self.dy_args, f"{tool_args.plan_dataf}/dy_model.pth")

    @profile
    def plan(
        self,
        state_cur,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        super().plan(state_cur, target_shape, rollout_path)

        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        max_bounds = torch.max(state_cur, dim=1).values.squeeze().cpu().numpy()
        self.param_bounds = get_param_bounds(
            self.dy_args, self.tool_params, min_bounds, max_bounds
        )

        best_loss = float("inf")
        for i in range(max_n_actions):
            pred_idx, param_seq, loss, state_seq = self.take_initial_guess(
                i, state_cur, target_shape
            )
            if self.optim_algo == "RS":
                param_seq_opt = param_seq
                loss_opt = loss
            elif self.optim_algo == "CEM":
                param_seq_CEM, loss_seq_CEM = self.optimize_CEM(
                    (param_seq, loss, state_seq), state_cur, target_shape
                )
                param_seq_opt = param_seq_CEM
                loss_opt = loss_seq_CEM
            elif self.optim_algo == "GD":
                with torch.set_grad_enabled(True):
                    param_seq_GD, loss_seq_GD = self.optimize_GD(
                        param_seq, state_cur, target_shape
                    )
                param_seq_opt = param_seq_GD
                loss_opt = loss_seq_GD
            else:
                raise NotImplementedError

            if loss_opt < best_loss:
                best_loss = loss_opt
                best_param_seq = param_seq_opt

        return best_param_seq

    def eval_soln(self, param_seq, state_cur, target_shape):
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        init_pose_seq = params_to_init_pose(
            self.dy_args, self.center, self.tool_params, param_seq
        ).numpy()
        act_seq_dense = params_to_actions(
            self.dy_args, self.tool_params, param_seq, min_bounds, step=1
        ).numpy()
        act_seq = params_to_actions(
            self.dy_args,
            self.tool_params,
            param_seq,
            min_bounds,
            step=self.dy_args.time_step,
        ).numpy()

        with torch.no_grad():
            state_pred_seq, _, _ = self.gnn.rollout(
                copy.deepcopy(state_cur), init_pose_seq[None], act_seq_dense[None]
            )

        state_cur_wshape = np.concatenate(
            (
                state_cur.cpu().numpy()[0, :, :3],
                self.dy_args.floor_state,
                init_pose_seq[0],
            ),
            axis=0,
        )
        state_seq_wshape = add_shape_to_seq(
            self.dy_args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq
        )
        state_seq_wshape = np.concatenate(
            (state_cur_wshape[None], state_seq_wshape), axis=0
        )

        state_out_pred = torch.tensor(
            state_seq_wshape[-1, : self.dy_args.n_particles], device=self.device
        )
        state_out = torch.tensor(target_shape["surf"], device=self.device)

        state_pred_norm = state_out_pred - torch.mean(state_out_pred, dim=0)
        state_goal_norm = state_out - torch.mean(state_out, dim=0)
        loss_gnn = chamfer(state_pred_norm, state_goal_norm, pkg="torch").item()

        title = f"{self.dy_args.env.upper()}: "
        for i in range(0, len(param_seq), 2):
            if i > 0:
                title += "\n"
            title += f"{[[round(x.item(), 3) for x in y] for y in param_seq[i:i+2]]}"
        title += f" -> {round(loss_gnn, 4)}"

        pkl_path = os.path.join(
            self.rollout_path,
            "anim_args",
            f'{self.dy_args.env}_anim_{datetime.now().strftime("%b-%d-%H:%M:%S")}_args.pkl',
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "args": self.dy_args,
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

        # render_anim(self.dy_args, [title], [state_seq_wshape], target=target_shape['sparse'],
        #     attn_mask_pred=[attn_mask_pred], rels_pred=[rels_pred],
        #     path=os.path.join(self.rollout_path, 'anim', f'{anim_name}.mp4'))

        with open(
            os.path.join(
                self.rollout_path, "param_seqs", f"{self.dy_args.env}_param_seq.yml"
            ),
            "w",
        ) as f:
            yaml.dump(
                {self.dy_args.env: param_seq.tolist()}, f, default_flow_style=True
            )

        info_dict = {
            "loss": [loss_gnn],
            "subprocess": [p],
        }

        return state_seq_wshape, info_dict

    @profile
    def take_initial_guess(self, idx, state_cur, target_shape):
        args = self.planner_dict["args"][idx]
        planner = self.planner_dict["model"][idx]
        state_in = copy.deepcopy(state_cur)
        state_out_ori = torch.tensor(
            target_shape["surf"], device=self.device
        ).unsqueeze(0)
        state_out = copy.deepcopy(state_out_ori)

        if args.use_normals:
            state_in_normals = get_normals(state_in, pkg="torch").to(self.device)
            state_in = torch.cat((state_in, state_in_normals), dim=-1)
            state_out_normals = get_normals(state_out, pkg="torch").to(self.device)
            state_out = torch.cat((state_out, state_out_normals), dim=-1)

        state_in_center = torch.mean(state_in[:, :, :3], dim=1)
        state_in[:, :, :3] = state_in[:, :, :3] - state_in_center
        state_in[:, :, :3] = state_in[:, :, :3] / args.scale_ratio

        state_out_center = torch.mean(state_out[:, :, :3], dim=1)
        state_out[:, :, :3] = state_out[:, :, :3] - state_out_center
        state_out[:, :, :3] = state_out[:, :, :3] / args.scale_ratio

        points = torch.cat((state_in, state_out), dim=1).to(torch.float32)

        if args.early_fusion:
            point_labels = torch.cat(
                (
                    torch.zeros((*state_in.shape[:-1], 1), device=self.device),
                    torch.ones((*state_out.shape[:-1], 1), device=self.device),
                ),
                dim=1,
            )
            points = torch.cat((points, point_labels), dim=-1)

        points = points.transpose(2, 1)

        pred, trans_feat = planner(points)

        pred_idx = [x[0].softmax(dim=0).argmax(dim=0).item() for x in pred[0]]
        pred_offsets = [x[0].cpu().numpy() for x in pred[1]]
        pred_params = []
        for i in range(len(pred_idx)):
            pred_param = args.bin_centers[i][pred_idx[i]] + pred_offsets[i][pred_idx[i]]
            if i % (len(pred_idx) // args.n_actions) == args.rot_idx:
                while pred_param < args.tool_params["rot_range"][0]:
                    pred_param += args.tool_params["rot_scope"]
                    pred_params[-1] *= (-1) ** (args.tool_params["rot_scope"] // np.pi)
                while pred_param > args.tool_params["rot_range"][1]:
                    pred_param -= args.tool_params["rot_scope"]
                    pred_params[-1] *= (-1) ** (args.tool_params["rot_scope"] // np.pi)
            else:
                pred_param *= args.scale_ratio
            pred_params.append(pred_param)

        pred_params_tensor = torch.tensor(pred_params, dtype=torch.float32).reshape(
            (args.n_actions, -1)
        )

        # state_cur = copy.deepcopy(state_in_ori)
        # state_target = copy.deepcopy(state_out_ori)
        # min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        # init_pose_seq = params_to_init_pose(self.dy_args, self.center, self.tool_params, pred_params).numpy()
        # act_seq_dense = params_to_actions(self.dy_args, self.tool_params, pred_params, min_bounds, step=1).numpy()

        # with torch.no_grad():
        #     state_pred_seq, _, _ = self.gnn.rollout(copy.deepcopy(state_cur), init_pose_seq[None], act_seq_dense[None])

        # state_pred = state_pred_seq[0, -1]
        # state_pred_norm = state_pred - torch.mean(state_pred, dim=0)

        # state_goal = state_target[0]
        # state_goal_norm = state_goal - torch.mean(state_goal, dim=0)
        # loss_gnn = chamfer(state_pred_norm, state_goal_norm, pkg='torch').item()

        loss, state_seq = self.rollout(
            pred_params_tensor.unsqueeze(0), state_cur, target_shape
        )

        # render_frames(self.dy_args, ['Init', f'Pred: {[round(x, 3) for x in pred_params]}', 'Target'], [state_cur.cpu(), state_seq[:, -1].cpu(), state_out_ori.cpu()],
        #     res='low', axis_off=False, focus=False, path=os.path.join(self.rollout_path, 'anim'),
        #     name=f'{self.dy_args.env}_action_{idx+1}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

        return pred_idx, pred_params_tensor, loss, state_seq[0]

    @profile
    def rollout(self, param_seqs, state_cur, target_shape):
        B = self.batch_size
        min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
        init_pose_seqs = []
        act_seqs = []
        for param_seq in param_seqs:
            init_pose_seq = params_to_init_pose(
                self.dy_args, self.center, self.tool_params, param_seq
            )
            act_seq = params_to_actions(
                self.dy_args, self.tool_params, param_seq, min_bounds
            )
            init_pose_seqs.append(init_pose_seq)
            act_seqs.append(act_seq)

        init_pose_seqs = torch.stack(init_pose_seqs)
        act_seqs = torch.stack(act_seqs)

        n_batch = int(np.ceil(param_seqs.shape[0] / B))
        state_seqs = []
        loss_seqs = []
        for i in range(n_batch):
            start = B * i
            end = min(B * (i + 1), param_seqs.shape[0])
            state_seq, _, _ = self.gnn.rollout(
                state_cur, init_pose_seqs[start:end], act_seqs[start:end]
            )
            loss_seq = self.evaluate_traj(state_seq, target_shape, "chamfer")
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
        loss_seq = []
        for i in range(state_seqs.shape[0]):
            state_cur = state_seqs[i, state_seqs.shape[1] - 1].unsqueeze(0)
            state_goal = torch.tensor(
                target_shape["surf"], device=self.device, dtype=torch.float32
            ).unsqueeze(0)

            state_cur_norm = state_cur - torch.mean(state_cur, dim=1)
            state_goal_norm = state_goal - torch.mean(state_goal, dim=1)

            # render_frames(self.dy_args, ['Current', 'Target'], [state_cur_norm.cpu(), state_goal_norm.cpu()],
            #     res='low', axis_off=False, focus=False)

            if loss_type == "chamfer":
                loss = self.chamfer_loss(state_cur_norm, state_goal_norm).cpu()
            elif loss_type == "emd":
                loss = self.emd_loss(state_cur_norm, state_goal_norm).cpu()
            elif loss_type == "chamfer_emd":
                chamfer_loss = self.chamfer_loss(state_cur_norm, state_goal_norm).cpu()
                emd_loss = self.emd_loss(state_cur_norm, state_goal_norm).cpu()
                loss = 0.5 * chamfer_loss + 0.5 * emd_loss
            else:
                raise NotImplementedError

            loss_seq.append(loss)

        loss_seq = torch.stack(loss_seq).cpu()

        return loss_seq

    @profile
    def optimize_CEM(
        self,
        soln,
        state_cur,
        target_shape,
        first_iter_threshold=5e-4,
        plateau_threshold=3e-4,
        plateau_max_iter=5,
    ):
        # https://github.com/martius-lab/iCEM

        param_seq_cur, loss_cur, state_seq_cur = soln
        n_actions = param_seq_cur.shape[0]
        best_param_seq, best_loss, best_state_seq = (
            param_seq_cur,
            loss_cur,
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

            param_seq_CEM = torch.tensor(param_seq_samples, dtype=torch.float32).view(
                sample_size, n_actions, -1
            )
            param_seq_CEM = torch.clamp(
                param_seq_CEM, min=self.param_bounds[:, 0], max=self.param_bounds[:, 1]
            )
            loss_list, state_seq_list = self.rollout(
                param_seq_CEM, state_cur, target_shape
            )

            sort_ind = torch.argsort(loss_list)
            traj_loss_list.append(loss_list[sort_ind[0]])

            param_seq_CEM = param_seq_CEM.reshape(sample_size, -1)
            param_seq_CEM_elite = param_seq_CEM[sort_ind[: self.CEM_elite_size]].numpy()
            param_seq_mean = np.mean(param_seq_CEM_elite, axis=0)
            param_seq_var = np.var(param_seq_CEM_elite, axis=0)

            sample_size = max(
                2 * self.CEM_elite_size, int(self.CEM_decay_factor * sample_size)
            )

            # check the improvement of absolute scale
            if j == 0 and best_loss - loss_list[sort_ind[0]] < first_iter_threshold:
                interrupt = f"---> Stop criterion 1: < {first_iter_threshold} improvement in the first iteration!"

            if best_loss - loss_list[sort_ind[0]] < plateau_threshold:
                plateau_iter += 1
            else:
                plateau_iter = 0

            if plateau_iter >= plateau_max_iter:
                interrupt = (
                    f"---> Stop criterion 2.b: improving less than {plateau_threshold} "
                    + f"for {plateau_max_iter} iterations!"
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

        param_seq_opt = best_param_seq.reshape(n_actions, -1)
        loss_opt = best_loss
        state_seq_opt = best_state_seq

        traj_loss_dict = {}
        traj_loss_dict["Traj 0"] = traj_loss_list

        # loss_opt = torch.tensor(loss_opt, dtype=torch.float32)

        plot_eval_loss(
            "Control CEM Loss",
            traj_loss_dict,
            path=os.path.join(
                self.rollout_path, "optim_plots", f"{self.dy_args.env}_CEM_loss"
            ),
        )

        # self.render_selections('CEM', state_cur, [param_seq_opt], [loss_opt], [state_seq_opt], target_shape)

        return param_seq_opt, loss_opt

    def optimize_GD(self, param_seq_cur, state_cur, target_shape):
        from torchmin import minimize

        param_seq_cand = param_seq_cur.detach().requires_grad_()

        param_seq_opt = []
        loss_list_dict = {}

        loss_list = []

        def loss_func(param_seq):
            nonlocal loss_list
            loss, _ = self.rollout(param_seq.unsqueeze(0), state_cur, target_shape)
            loss_list.append(loss[0].item())
            return loss[0]

        # import pdb; pdb.set_trace()
        res = minimize(
            loss_func,
            param_seq_cand.contiguous(),
            method="l-bfgs",
            options=dict(lr=1e-1, line_search="strong-wolfe"),
            disp=2,
        )

        param_seq_opt = res.x
        loss_opt = res.fun

        loss_list_dict[f"Traj 0"] = loss_list

        plot_eval_loss(
            "Control GD Loss",
            loss_list_dict,
            path=os.path.join(
                self.rollout_path, "optim_plots", f"{self.dy_args.env}_GD_loss"
            ),
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
                self.dy_args, self.center, self.tool_params, param_seq_opt[i]
            )
            act_seq = params_to_actions(
                self.dy_args,
                self.tool_params,
                param_seq_opt[i],
                min_bounds,
                step=self.dy_args.time_step,
            )
            state_seq_wshape = add_shape_to_seq(
                self.dy_args,
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
            f'{self.dy_args.env}_anim_{name}_{datetime.now().strftime("%b-%d-%H:%M:%S")}_args.pkl',
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "args": self.dy_args,
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

        # render_anim(self.dy_args, row_titles, state_seq_wshape_list, target=target_shape['sparse'],
        #     path=os.path.join(self.rollout_path, 'anim', f'{self.dy_args.env}_anim_{name}.mp4'))
