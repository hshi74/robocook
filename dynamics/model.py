import pdb
import numpy as np
import scipy
import torch

torch.set_default_tensor_type(torch.FloatTensor)

import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import quaternion_to_matrix
from scipy import optimize
from torch.autograd import Variable
from utils.data_utils import *

# from utils.Density_aware_Chamfer_Distance.utils_v2.model_utils import calc_dcd


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        s_x = x.size()

        if self.residual:
            s_res = res.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if self.residual:
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [-1])
        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [-1])


class AttnLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttnLayer, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.sigmoid(self.linear(x))

        return x.view(list(s_x[:-1]) + [-1])


class DynamicsPredictor(nn.Module):
    def __init__(self, args, residual=False):
        super(DynamicsPredictor, self).__init__()

        self.args = args

        self.n_his = args.n_his
        self.time_step = args.time_step
        self.attr_dim = 4 if "gripper_asym" in args.env else 3
        self.state_dim = args.state_dim
        self.mem_dim = args.nf_effect * args.mem_nlayer

        self.device = args.device
        self.residual = residual

        self.quat_offset = torch.tensor([1, 0, 0, 0], device=self.device)

        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        # ParticleEncoder
        particle_input_dim = (
            self.attr_dim
            + 1
            + (self.n_his * self.time_step + 1) * self.state_dim
            + self.mem_dim
        )
        self.particle_encoder = Encoder(particle_input_dim, nf_particle, nf_effect)

        # RelationEncoder
        relation_input_dim = (
            particle_input_dim * 2
            + 1
            + (self.n_his * self.time_step + 1) * self.state_dim
        )
        self.relation_encoder = Encoder(relation_input_dim, nf_relation, nf_effect)

        # ParticlePropagator
        self.particle_propagator = Propagator(nf_effect * 2, nf_effect, self.residual)

        # RelationPropagator
        self.relation_propagator = Propagator(nf_effect * 3, nf_effect)

        # ParticlePredictor
        self.rigid_predictor = ParticlePredictor(nf_effect, nf_effect, 7)
        self.non_rigid_predictor = ParticlePredictor(nf_effect, nf_effect, 3)

        self.nrm_attn_layer = AttnLayer(nf_effect, 1)
        # self.rm_attn_layer = AttnLayer(nf_effect, 1)

    # @profile
    def forward(self, inputs):
        args = self.args

        # attrs: B x N x attr_dim
        # state (unnormalized): B x n_his x N x state_dim
        # Rr_cur, Rs_cur: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # group:
        #   p_rigid: B x n_instance
        #   p_instance: B x n_particle x n_instance
        #   physics_param: B x n_particle
        attrs, state_cur, action_his, Rr_cur, Rs_cur, memory, group, stats = inputs
        p_rigid, p_instance, physics_param = group
        mean_p, std_p, mean_d, std_d = stats

        # B: batch_size
        # N: the number of all particles
        # n_p: the number of particles of the deformable object
        # n_s: the number of particles of other objects (floor and tools)
        # n_his: history length
        # state_dim: 3
        B, N = attrs.size(0), attrs.size(1)
        n_p = p_instance.size(1)
        n_s = attrs.size(1) - n_p

        # Rr_cur_t, Rs_cur_t: B x N x n_rel
        Rr_cur_t = Rr_cur.transpose(1, 2).contiguous()
        # Rs_cur_t = Rs_cur.transpose(1, 2).contiguous()
        p_instance_t = p_instance.transpose(1, 2)

        # state_norm (normalized): B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        state_res_norm = torch.zeros(
            B, self.n_his * self.time_step, N, self.state_dim, device=self.device
        )
        state_res_norm[:, :, n_p + args.floor_dim :, :3] = batch_normalize(
            action_his[:, 1:, :, :3] - action_his[:, :-1, :, :3], mean_d, std_d
        )

        # [n_his - 1, n_his): the current position
        state_cur_norm = batch_normalize(state_cur.unsqueeze(1), mean_p, std_p)
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        # state_norm_t (normalized): B x N x (n_his * state_dim)
        state_norm_t = (
            state_norm.transpose(1, 2)
            .contiguous()
            .view(B, N, (self.n_his * self.time_step + 1) * self.state_dim)
        )

        # add offset to center-of-mass for rigids to attr
        # offset: B x N x (n_his * state_dim)
        offset = torch.zeros(
            B, N, (self.n_his * self.time_step + 1) * self.state_dim, device=self.device
        )

        # p_rigid_per_p: B x n_p x 1
        # this is trying to keep both instance label and rigidity label
        p_rigid_per_p = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)

        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance_t.bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + 1e-6

        # c_per_p: B x n_p x (n_his * state_dim)
        # particle offset: B x n_p x (n_his * state_dim)
        c_per_p = p_instance.bmm(instance_center)
        c = (1 - p_rigid_per_p) * state_norm_t[:, :n_p] + p_rigid_per_p * c_per_p
        offset[:, :n_p] = state_norm_t[:, :n_p] - c

        # memory_t: B x N x (mem_nlayer * nf_memory)
        # physics_param: B x N x 1
        # attrs: B x N x (attr_dim + 1 + n_his * state_dim + mem_nlayer * nf_memory)
        memory_t = memory.transpose(1, 2).contiguous().view(B, N, -1)
        physics_param_s = torch.zeros(B, n_s, 1, device=self.device)
        physics_param = torch.cat([physics_param[:, :, None], physics_param_s], 1)
        attrs = torch.cat([attrs, physics_param, offset, memory_t], 2)

        # group info
        # g: B x N x n_instance
        g = p_instance
        g_s = torch.zeros(B, n_s, args.n_instance, device=self.device)
        g = torch.cat([g, g_s], 1)

        # receiver_attr, sender_attr
        # attrs_r: B x n_rel x -1
        # attrs_s: B x n_rel x -1
        attrs_r = Rr_cur.bmm(attrs)
        attrs_s = Rs_cur.bmm(attrs)
        # receiver_state, sender_state
        # state_norm_r: B x n_rel x -1
        # state_norm_s: B x n_rel x -1
        state_norm_r = Rr_cur.bmm(state_norm_t)
        state_norm_s = Rs_cur.bmm(state_norm_t)

        # receiver_group, sender_group
        # group_r: B x n_rel x -1
        # group_s: B x n_rel x -1
        group_r = Rr_cur.bmm(g)
        group_s = Rs_cur.bmm(g)
        group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

        # calculate particle encoding
        # particle_encode = self.particle_encoder(torch.cat([attrs, state_norm_t], 2))
        particle_encode = self.particle_encoder(attrs)
        particle_effect = particle_encode

        # calculate relation encoding
        # relation_encode = self.relation_encoder(
        #     torch.cat([attrs_r, attrs_s, state_norm_r, state_norm_s, group_diff], 2))
        relation_encode = self.relation_encoder(
            torch.cat([attrs_r, attrs_s, state_norm_r - state_norm_s, group_diff], 2)
        )

        for i in range(args.pstep):
            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr_cur.bmm(particle_effect)
            effect_s = Rs_cur.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)
            )

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_cur_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2), res=particle_effect
            )

        # non_rigid_motion: B x n_p x state_dim
        non_rigid_motion = self.non_rigid_predictor(
            particle_effect[:, :n_p].contiguous()
        )

        # rigid motion
        # instance effect: B x n_instance x nf_effect
        n_instance = p_instance.size(2)
        instance_effect = p_instance_t.bmm(particle_effect[:, :n_p])

        # rigid motion
        # instance_rigid_params: (B * n_instance) x 7
        instance_rigid_params = self.rigid_predictor(instance_effect).view(
            B * n_instance, 7
        )

        # R: (B * n_instance) x 3 x 3
        R = quaternion_to_matrix(instance_rigid_params[:, :4] + self.quat_offset)

        b = batch_denormalize(instance_rigid_params[:, 4:], mean_d, std_d)
        b = b.view(B * n_instance, 1, 3)

        p_0 = state_cur[:, None, :n_p, :3]
        p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, 3)

        c = batch_denormalize(instance_center[:, :, -3:], mean_p[:3], std_p[:3])
        c = c.view(B * n_instance, 1, 3)

        p_1 = torch.bmm(p_0 - c, R) + b + c

        # rigid_motion: B x n_instance x n_p x state_dim
        rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, 3)
        rigid_motion = batch_normalize(rigid_motion, mean_d, std_d)

        nrm_attn = torch.zeros(B, n_p, 1, device=self.device)
        # rm_attn = torch.zeros(B, n_p, 1, device=self.device)
        if args.rigid_motion:
            rigid_motion = torch.sum(p_instance_t[:, :, :, None] * rigid_motion, 1)
            if args.attn:
                nrm_attn = self.nrm_attn_layer(particle_effect[:, :n_p].contiguous())
                # rm_attn = self.rm_attn_layer(instance_effect).repeat(1, rigid_motion.shape[1], 1)
                # import pdb; pdb.set_trace()
                motion_norm = nrm_attn * non_rigid_motion + rigid_motion
                # motion_norm = (non_rigid_motion + rm_attn * rigid_motion)
            else:
                motion_norm = non_rigid_motion + rigid_motion
        else:
            # if args.attn:
            #     nrm_attn = self.nrm_attn_layer(particle_effect[:, :n_p].contiguous())
            #     motion_norm = (nrm_attn * non_rigid_motion)
            # else:
            motion_norm = non_rigid_motion

        is_tool_touching = torch.count_nonzero(
            Rs_cur[:, :, n_p + args.floor_dim :], dim=(1, 2)
        )
        pred_motion = (is_tool_touching > 0).int().view(-1, 1, 1) * batch_denormalize(
            motion_norm, mean_d, std_d
        )
        if args.loss_type == "chamfer_emd":
            pred_pos = state_cur[:, :n_p, :3] + torch.clamp(
                pred_motion,
                max=self.time_step * args.motion_bound,
                min=-self.time_step * args.motion_bound,
            )
        else:
            pred_pos = state_cur[:, :n_p, :3] + pred_motion

        # pred_pos (unnormalized): B x n_p x state_dim
        # pred_motion_norm (normalized): B x n_p x state_dim
        return pred_pos, pred_motion, nrm_attn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(
            B, self.args.mem_nlayer, N, self.args.nf_effect, device=self.device
        )

        return mem

    def predict_dynamics(self, inputs):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(inputs)
        return ret


class ChamferLoss(torch.nn.Module):
    def __init__(self, ord):
        super(ChamferLoss, self).__init__()
        self.ord = ord

    def chamfer_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), self.ord, dim=3)  # dis: [B, N, M]
        dis_xy = torch.mean(torch.min(dis, dim=2)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=1)[0])  # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.chamfer_distance(pred, label)


class EarthMoverLoss(torch.nn.Module):
    def __init__(self, ord):
        super(EarthMoverLoss, self).__init__()
        self.ord = ord

    # @profile
    def em_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        # x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
        # y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
        # dis = torch.norm(torch.add(x_, -y_), self.ord, dim=3)  # dis: [B, N, M]
        x_ = x.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        x_list = []
        y_list = []
        # x.requires_grad = True
        # y.requires_grad = True
        for i in range(x.shape[0]):
            cost_matrix = scipy.spatial.distance.cdist(x_[i], y_[i])
            # cost_matrix = dis[i].detach().cpu().numpy()
            try:
                ind1, ind2 = scipy.optimize.linear_sum_assignment(
                    cost_matrix, maximize=False
                )
            except:
                # pdb.set_trace()
                print("Error in linear sum assignment!")
            x_list.append(x[i, ind1])
            y_list.append(y[i, ind2])
            # x[i] = x[i, ind1]
            # y[i] = y[i, ind2]
        new_x = torch.stack(x_list)
        new_y = torch.stack(y_list)
        # print(f"EMD new_x shape: {new_x.shape}")
        # print(f"MAX: {torch.max(torch.norm(torch.add(new_x, -new_y), 2, dim=2))}")
        emd = torch.mean(torch.norm(torch.add(new_x, -new_y), self.ord, dim=2))

        return emd

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.em_distance(pred, label)


class HausdorffLoss(torch.nn.Module):
    def __init__(self, ord):
        super(HausdorffLoss, self).__init__()
        self.ord = ord

    def hausdorff_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), self.ord, dim=3)  # dis: [B, N, M]
        # print(dis.shape)
        dis_xy = torch.max(torch.min(dis, dim=2)[0])  # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=1)[0])  # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.hausdorff_distance(pred, label)


# class DCDLoss(torch.nn.Module):
#     def __init__(self, alpha, n_lambda):
#         super(DCDLoss, self).__init__()
#         self.alpha = alpha
#         self.n_lambda = n_lambda

#     def dc_distance(self, x, y):
#         # x: [B, N, D]
#         # y: [B, M, D]
#         return torch.mean(calc_dcd(x, y, alpha=self.alpha, n_lambda=self.n_lambda)[0])

#     def __call__(self, pred, label):
#         # pred: [B, N, D]
#         # label: [B, M, D]
#         return self.dc_distance(pred, label)
