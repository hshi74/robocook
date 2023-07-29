import gym
import numpy as np
import os
import torch
import yaml

from ..config import load
from dynamics.model import EarthMoverLoss, ChamferLoss, HausdorffLoss
from dynamics.dy_utils import prepare_input
from gym.spaces import Box
from .utils import merge_lists
from utils.data_utils import load_data, get_scene_info, get_env_group
from yacs.config import CfgNode


PATH = os.path.dirname(os.path.abspath(__file__))

##### DEPRECATED AND WON'T WORK #####
class PlasticineEnv(gym.Env):
    def __init__(self, cfg_path, version, args=None, nn=False, learned_model=None, use_gpu=True, device=None):
        from ..engine.taichi_env import TaichiEnv
        self.args = args
        self.cfg_path = cfg_path
        cfg = self.load_varaints(cfg_path, version)
        self.use_gpu = use_gpu
        self.device = device
        self.taichi_env = TaichiEnv(cfg, nn, loss=False) # build taichi environment
        self.taichi_env.initialize()
        self.cfg = cfg.ENV
        self.taichi_env.set_copy(True)
        self._init_state = self.taichi_env.get_state()
        self.taichi_env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
        self.taichi_env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])
        def set_parameters(env: TaichiEnv, yield_stress, E, nu):
            env.simulator.yield_stress.fill(yield_stress)
            _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
            env.simulator.mu.fill(_mu)
            env.simulator.lam.fill(_lam)
        set_parameters(self.taichi_env, yield_stress=200, E=5e3, nu=0.2)
        self._n_observed_particles = self.cfg.n_observed_particles
        self.learned_model = learned_model
        self.n_particle = 300
        self.n_shape = 31
        self.loaddata_init()
        self.emd_loss = EarthMoverLoss()
        self.chamfer_loss = ChamferLoss()
        self.h_loss = HausdorffLoss()
        obs = self.resetm()
        if 'gripper' in cfg_path and learned_model:
            self.observation_space = Box(-np.inf, np.inf, (self.n_particle*3, ))
            self.action_space = Box(np.array([-.1, -.1, 0, self.task_params['gripper_rate_limits'][0]]),
                                    np.array([.1, .1, np.pi, self.task_params['gripper_rate_limits'][1]]),
                                    (4,))
        else:
            self.observation_space = Box(-np.inf, np.inf, obs.shape)
            self.action_space = Box(-1, 1, (self.taichi_env.primitives.action_dim,))


    def loaddata_init(self):
        self.task_params = {
            "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
            "sample_radius": 0.4,
            "len_per_grip": 30,
            "len_per_grip_back": 10,
            "floor_pos": np.array([0.5, 0, 0.5]),
            "n_shapes": 3,
            "n_shapes_floor": 9,
            "n_shapes_per_gripper": 11,
            "gripper_mid_pt": int((11 - 1) / 2),
            "gripper_gap_limits": np.array([0.14, 0.06]),
            "p_noise_scale": 0.01,
            "p_noise_bound": 0.03,
            "loss_weights": [0.3, 0.7, 0.1, 0.0],
            "tool_size": 0.045
        }
        self.task_params["gripper_rate_limits"] = [
            (self.task_params['sample_radius'] * 2 - (
            self.task_params['gripper_gap_limits'][0] + 2 * self.task_params['tool_size'])) / (2 * self.task_params['len_per_grip']),
            (self.task_params['sample_radius'] * 2 - (
            self.task_params['gripper_gap_limits'][1] + 2 * self.task_params['tool_size'])) / (2 * self.task_params['len_per_grip'])
        ]
        task_name = 'gripper'
        data_names = ['positions', 'shape_quats', 'scene_params']
        # --- please add your path to dynamics here --- #
        rollout_dir = f"../dynamics/data/data_{self.args.task_type}/train/"

        if task_name == "gripper":
            frame_name = str(0) + '.h5'
            frame_name = 'shape_' + frame_name
            frame_path = os.path.join(rollout_dir, str(0).zfill(3), frame_name)
        else:
            raise NotImplementedError
        this_data = load_data(data_names, frame_path)

        n_particle, n_shape, scene_params = get_scene_info(this_data)
        self.scene_params = torch.FloatTensor(scene_params).unsqueeze(0)
        self.pstates_init = this_data[0]
        self.sstates_init = this_data[1]
        self.cur_pstates = this_data[0]
        ### load goal shape
        if len(self.args.target_shape_name) > 0 and \
            self.args.target_shape_name != 'none' and \
            self.args.target_shape_name[:3] != 'vid':
            # --- load your path to dynamics here ---#
            shape_dir = os.path.join('../dynamics', 'shapes', self.args.target_shape_name)
            self.goal_shapes = []
            for i in range(self.args.n_grips):
                goal_frame_name = f'{self.args.target_shape_name}.h5'
                goal_frame_path = os.path.join(shape_dir, goal_frame_name)
                goal_data = load_data(data_names, goal_frame_path)
                self.goal_shapes.append(torch.FloatTensor(goal_data[0]).unsqueeze(0)[:, :n_particle, :])

    def resetm(self, evaluation=False):
        if evaluation:
            self.taichi_env.set_state(**self._init_state)
            self.cur_pstates = self.pstates_init[:self.n_particle, :]
        else:
            self.cur_pstates = self.pstates_init[:self.n_particle, :]
        self.cur_distance = self.evaluate_traj(torch.FloatTensor(self.cur_pstates).unsqueeze(0), self.goal_shapes[0])
        self.cur_distance = self.cur_distance.numpy()
        return self.cur_pstates.reshape(self.n_particle*3)

    def reset(self):
        self.taichi_env.set_state(**self._init_state)
        self._recorded_actions = []
        return self._get_obs()

    def _get_obsm(self):
        return self.cur_pstates

    def _get_obs(self, t=0):
        x = self.taichi_env.simulator.get_x(t)
        v = self.taichi_env.simulator.get_v(t)
        outs = []
        for i in self.taichi_env.primitives:
            outs.append(i.get_state(t))
        s = np.concatenate(outs)
        step_size = len(x) // self._n_observed_particles
        return np.concatenate((np.concatenate((x[::step_size], v[::step_size]), axis=-1).reshape(-1), s.reshape(-1)))

    def stepm(self, pose_action, evaluation=False):
        # pose_action include p_noise_x, p_noise_z, rot_noise, gripper_rate
        p_noise_x = pose_action[0]
        p_noise_z = pose_action[1]
        rot_noise = pose_action[2]
        gripper_rate = pose_action[3]
        p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
        new_mid_point = self.task_params["mid_point"][:3] + p_noise
        init_pose = self.get_pose(new_mid_point, rot_noise)
        actions = self.get_action_seq(rot_noise, gripper_rate)
        reward_seqs, model_state_seqs=self.rollout(init_pose, actions, self.cur_pstates, self.goal_shapes[0], evaluation)
        self.cur_pstates = model_state_seqs[0, -1, :self.n_particle, :]
        reward =  self.cur_distance - reward_seqs[-1]
        self.cur_distance = reward_seqs[-1]
        return model_state_seqs[0, -1, :self.n_particle, :].reshape(self.n_particle*3),  reward, False, {}

    def step(self, action):
        self.taichi_env.step(action)
        loss_info = {'reward': 0}

        self._recorded_actions.append(action)
        obs = self._get_obs()
        r = loss_info['reward']
        if np.isnan(obs).any() or np.isnan(r):
            if np.isnan(r):
                print('nan in r')
            import pickle, datetime
            with open(f'{self.cfg_path}_nan_action_{str(datetime.datetime.now())}', 'wb') as f:
                pickle.dump(self._recorded_actions, f)
            raise Exception("NaN..")
        return obs, r, False, loss_info

    def render(self, mode='human'):
        return self.taichi_env.render(mode)

    def render_multi(self, mode='human'):
        return self.taichi_env.render_multi(mode)

    def rollout(self, init_pose, actions, state_cur, state_goal, evaluation):
        reward_seqs_rollout = []
        state_seqs_rollout = []
        if evaluation:
            state_seqs = self.sim_rollout(init_pose, actions)
        else:
            state_seqs = self.model_rollout(state_cur, init_pose, actions)
        reward_seqs = self.evaluate_traj(state_seqs[0], state_goal)

        reward_seqs_rollout.append(reward_seqs)
        state_seqs_rollout.append(state_seqs)

        reward_seqs_rollout = torch.cat(reward_seqs_rollout, 0)
        state_seqs_rollout = torch.cat(state_seqs_rollout, 0)

        return reward_seqs_rollout.detach().numpy(), state_seqs_rollout.detach().numpy()

    def sim_rollout(self, init_pose, actions):
        sample_state_seq_batch = []
        state_seq = []

        self.taichi_env.primitives.primitives[0].set_state(0, init_pose[5, :7])
        self.taichi_env.primitives.primitives[1].set_state(0, init_pose[5, 7:])
        for j in range(actions.shape[0]):
            self.taichi_env.step(actions[j])
            x = self.taichi_env.simulator.get_x(0)
            step_size = len(x) // self.n_particle
            x = x[::step_size]
            particles = x[:self.n_particle]
            state_seq.append(particles)
        state_seq_batch = torch.from_numpy(np.stack(state_seq))[None, ...]

        return state_seq_batch

    def model_rollout(
            self,
            state_cur,  # [1, n_his, state_dim]
            init_pose,
            actions,  # [n_sample, -1, action_dim]
    ):
        if not torch.is_tensor(init_pose):
            init_pose_seqs = torch.tensor(init_pose)

        if not torch.is_tensor(actions):
            act_seqs = torch.tensor(actions)

        if not torch.is_tensor(state_cur):
            state_cur = torch.tensor(state_cur)

        init_pose = init_pose.float()[None, None, ...].to(self.device)
        actions = actions.float()[None, ...].to(self.device)
        state_cur = state_cur.float()[None, None, ...].to(self.device)
        floor_state = torch.from_numpy(np.array([[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
                                [0.5,  0., 0.25], [0.5,  0., 0.5], [0.5,  0., 0.75],
                                [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]])).float()[None, None, ...].to(self.device)
        init_pose = init_pose.repeat(1, self.args.n_his, 1, 1)
        state_cur = state_cur.repeat(1, self.args.n_his, 1, 1)
        floor_state = floor_state.repeat(1, self.args.n_his, 1, 1)

        memory_init = self.learned_model.init_memory(1, self.n_particle + self.n_shape)
        scene_params = self.scene_params.expand(1, -1)
        group_gt = get_env_group(self.args, self.n_particle, scene_params, use_gpu=self.use_gpu)

        states_pred_list = []

        # the shape of init_pose is: #_of_particles x (2xdof): 11 x 14
        shape1 = init_pose[:,:,:, :3]
        shape2 = init_pose[:,:,:, 7:10]
        state_cur = torch.cat([state_cur[:, :, :self.n_particle, :], floor_state,
                               shape1,
                               shape2], dim=2)
        for j in range(actions.shape[1]):
            attrs = []
            Rr_curs = []
            Rs_curs = []
            Rn_curs = []
            max_n_rel = 0
            for k in range(actions.shape[0]):
                state_last = state_cur[k][-1]
                attr, _, Rr_cur, Rs_cur, Rn_cur = prepare_input(
                    self.args, state_last.detach().cpu().numpy()
                )
                attr = attr.to(self.device)
                Rr_cur = Rr_cur.to(self.device)
                Rs_cur = Rs_cur.to(self.device)
                Rn_cur = Rn_cur.to(self.device)
                max_n_rel = max(max_n_rel, Rr_cur.size(0))
                attr = attr.unsqueeze(0)
                Rr_cur = Rr_cur.unsqueeze(0)
                Rs_cur = Rs_cur.unsqueeze(0)
                Rn_cur = Rn_cur.unsqueeze(0)
                attrs.append(attr)
                Rr_curs.append(Rr_cur)
                Rs_curs.append(Rs_cur)
                Rn_curs.append(Rn_cur)

            attrs = torch.cat(attrs, dim=0)
            for k in range(len(Rr_curs)):
                Rr, Rs, Rn = Rr_curs[k], Rs_curs[k], Rn_curs[k]
                Rr = torch.cat(
                    [Rr, torch.zeros((1, max_n_rel - Rr.size(1), self.n_particle + self.n_shape)).to(self.device)], 1)
                Rs = torch.cat(
                    [Rs, torch.zeros((1, max_n_rel - Rs.size(1), self.n_particle + self.n_shape)).to(self.device)], 1)
                Rn = torch.cat(
                    [Rn, torch.zeros((1, max_n_rel - Rn.size(1), self.n_particle + self.n_shape)).to(self.device)], 1)
                Rr_curs[k], Rs_curs[k], Rn_curs[k] = Rr, Rs, Rn

            Rr_curs = torch.cat(Rr_curs, dim=0)
            Rs_curs = torch.cat(Rs_curs, dim=0)
            Rn_curs = torch.cat(Rn_curs, dim=0)

            inputs = [attrs, state_cur, Rr_curs, Rs_curs, Rn_curs, memory_init, group_gt, None]
            pred_pos, pred_motion_norm = self.learned_model.predict_dynamics(inputs)

            shape1 += actions[:, j, :3].unsqueeze(1).expand(-1,self.task_params["n_shapes_per_gripper"], -1) * 0.02
            shape2 += actions[:, j, 6:9].unsqueeze(1).expand(-1, self.task_params["n_shapes_per_gripper"], -1) * 0.02

            pred_pos = torch.cat(
                [pred_pos, state_cur[:, -1, self.n_particle: self.n_particle + self.task_params["n_shapes_floor"], :],
                 shape1[:, 0, :, :], shape2[:, 0, :, :]], 1)

            state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
            states_pred_list.append(pred_pos[:, :self.n_particle, :])

        states_pred_array = torch.stack(states_pred_list, dim=1).cpu()

        return states_pred_array

    def evaluate_traj(
            self,
            state_seqs,  # [n_sample, n_look_ahead, state_dim]
            state_goal,  # [state_dim]
    ):
        reward_seqs = []
        # 40 x 300 x 3 for state_seqs
        for i in range(state_seqs.shape[0]):
            state_final = state_seqs[i, ...].unsqueeze(0)
            if state_final.shape != state_goal.shape:
                print("Data shape doesn't match in evaluate_traj!")
                raise ValueError

            # smaller loss, larger reward
            if self.args.reward_type == "emd":
                loss = self.emd_loss(state_final, state_goal)
            elif self.args.reward_type == "chamfer":
                loss = self.chamfer_loss(state_final, state_goal)
            elif self.args.reward_type == "chamfer_emd":
                emd_weight, chamfer_weight = self.task_params["loss_weights"]
                loss = 0
                if emd_weight > 0:
                    emd = emd_weight * self.emd_loss(state_final, state_goal)
                    loss += emd
                    print(f'emd: {emd/emd_weight}')
                if chamfer_weight > 0:
                    chamfer = chamfer_weight * self.chamfer_loss(state_final, state_goal)
                    print(f'chamfer: {chamfer/chamfer_weight}')
                    loss += chamfer
            else:
                raise NotImplementedError

            reward_seqs.append(loss)

        reward_seqs = torch.stack(reward_seqs).detach()
        return reward_seqs

    def get_pose(self, new_mid_point, rot_noise):
        if not torch.is_tensor(rot_noise):
            rot_noise = torch.tensor(rot_noise)

        x1 = new_mid_point[0] - self.task_params["sample_radius"] * torch.cos(rot_noise)
        y1 = new_mid_point[2] + self.task_params["sample_radius"] * torch.sin(rot_noise)
        x2 = new_mid_point[0] + self.task_params["sample_radius"] * torch.cos(rot_noise)
        y2 = new_mid_point[2] - self.task_params["sample_radius"] * torch.sin(rot_noise)

        unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        new_prim1 = []
        for j in range(self.task_params["n_shapes_per_gripper"]):
            prim1_pos = torch.stack([x1, torch.tensor(new_mid_point[1].item() + 0.018 * (j - 5)), y1])
            prim1_tmp = torch.cat((prim1_pos, unit_quat))
            new_prim1.append(prim1_tmp)
        new_prim1 = torch.stack(new_prim1)

        new_prim2 = []
        for j in range(self.task_params["n_shapes_per_gripper"]):
            prim2_pos = torch.stack([x2, torch.tensor(new_mid_point[1].item() + 0.018 * (j - 5)), y2])
            prim2_tmp = torch.cat((prim2_pos, unit_quat))
            new_prim2.append(prim2_tmp)
        new_prim2 = torch.stack(new_prim2)

        init_pose = torch.cat((new_prim1, new_prim2), 1)

        return init_pose

    def get_action_seq(self, rot_noise, gripper_rate):
        if not torch.is_tensor(rot_noise):
            rot_noise = torch.tensor(rot_noise)

        if not torch.is_tensor(gripper_rate):
            gripper_rate = torch.tensor(gripper_rate)

        zero_pad = torch.zeros(3)
        actions = []
        counter = 0
        while counter < self.task_params["len_per_grip"]:
            x = gripper_rate * torch.cos(rot_noise)
            y = -gripper_rate * torch.sin(rot_noise)
            prim1_act = torch.stack([x / 0.02, torch.tensor(0), y / 0.02])
            prim2_act = torch.stack([-x / 0.02, torch.tensor(0), -y / 0.02])
            act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
            actions.append(act)
            counter += 1

        counter = 0
        while counter < self.task_params["len_per_grip_back"]:
            x = -gripper_rate * torch.cos(rot_noise)
            y = gripper_rate * torch.sin(rot_noise)
            prim1_act = torch.stack([x / 0.02, torch.tensor(0), y / 0.02])
            prim2_act = torch.stack([-x / 0.02, torch.tensor(0), -y / 0.02])
            act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
            actions.append(act)
            counter += 1

        actions = torch.stack(actions)
        return actions

    @classmethod
    def load_varaints(self, cfg_path, version):
        assert version >= 1
        cfg_path = os.path.join(PATH, cfg_path)
        cfg = load(cfg_path)
        variants = cfg.VARIANTS[version - 1]

        new_cfg = CfgNode(new_allowed=True)
        new_cfg = new_cfg._load_cfg_from_yaml_str(yaml.safe_dump(variants))
        new_cfg.defrost()
        if 'PRIMITIVES' in new_cfg:
            new_cfg.PRIMITIVES = merge_lists(cfg.PRIMITIVES, new_cfg.PRIMITIVES)
        if 'SHAPES' in new_cfg:
            new_cfg.SHAPES = merge_lists(cfg.SHAPES, new_cfg.SHAPES)
        cfg.merge_from_other_cfg(new_cfg)

        cfg.defrost()
        # set target path id according to version
        name = list(cfg.ENV.loss.target_path)
        name[-5] = str(version)
        cfg.ENV.loss.target_path = os.path.join(PATH, '../', ''.join(name))
        cfg.VARIANTS = None
        cfg.freeze()

        return cfg
