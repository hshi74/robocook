import gc
import glob
import numpy as np
import os
import torch
import yaml

from planning.control_utils import *
from dynamics.gnn import GNN
from tqdm import tqdm
from utils.config import *
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_params(args, center, state_in, state_out):
    if 'gripper' in args.env:
        in_tool_center_list = []
        out_tool_center_list = []
        tool_start = args.n_particles + args.floor_dim
        for i in range(len(args.tool_dim[args.env])):
            tool_dim = args.tool_dim[args.env][i]
            in_tool_center_list.append(np.mean(state_in[tool_start:tool_start+tool_dim, :2], axis=0))
            out_tool_center_list.append(np.mean(state_out[tool_start:tool_start+tool_dim, :2], axis=0))
            tool_start += tool_dim

        in_tool_center = np.stack(in_tool_center_list)
        out_tool_center = np.stack(out_tool_center_list)

        in_tool_vec = in_tool_center[1] - in_tool_center[0]
        dist = np.abs(np.cross(in_tool_vec, in_tool_center[0] - center)) \
            / np.linalg.norm(in_tool_vec)
        rot = np.arctan2(in_tool_vec[1], in_tool_vec[0]) - 3 * np.pi / 4

        # in_center_dist = np.linalg.norm(in_tool_center[1] - in_tool_center[0])
        out_center_dist = np.linalg.norm(out_tool_center[1] - out_tool_center[0])

        tool_center_l = abs(np.mean(args.tool_full_repr_dict[args.env][0], axis=0)[0])
        tool_center_r = abs(np.mean(args.tool_full_repr_dict[args.env][1], axis=0)[0])
        grip_width = out_center_dist - tool_center_l - tool_center_r

        return np.array([dist, rot, grip_width])


# @profile
def build_planning_dataset(args, tool_name, tool_model_path, max_n_actions=2, B=8):
    print(tool_name, tool_model_path)
    tool_type = tool_model_path.split('/')[-2].replace('dump_', '')
    dy_dataset_root = os.path.join('/scr/hshi74/projects/robocook/data', 'keyframe', f'data_{tool_type}')

    tool_args = copy.deepcopy(args)
    tool_args.env = tool_name
    
    # dy_args_dict = np.load(f'{tool_model_path}_args.npy', allow_pickle=True).item()
    dy_args_dict = np.load(f'{tool_model_path}/args.npy', allow_pickle=True).item()
    tool_args = update_dy_args(tool_args, tool_name, dy_args_dict)

    with open('/scr/hshi74/projects/robocook/config/tool_plan_params.yml', 'r') as f:
        tool_params = yaml.load(f, Loader=yaml.FullLoader)[tool_args.env]

    sample_size = 128
    if 'gripper' in args.env:
        sample_ratio = (4, 8, 4)
    elif 'press' in args.env or 'punch' in args.env:
        if 'circle' in args.env:
            sample_ratio = (4, 8, 4)
        else:
            sample_ratio = (4, 4, 4, 2)
    elif 'roller' in args.env:
        sample_ratio = (2, 2, 4, 4, 2)
    else:
        raise NotImplementedError

    shape_quats = np.zeros((sum(tool_args.tool_dim[tool_args.env]) + tool_args.floor_dim, 4), dtype=np.float32)
    
    # print_args(tool_args)
    # gnn = GNN(tool_args, f'{tool_model_path}.pth')
    gnn = GNN(tool_args, f'{tool_model_path}/net_best.pth')

    if not tool_name in tool_type:
        tool_type = tool_type.replace('square', 'circle')
    plan_dataset_root = os.path.join('/scr/hshi74/projects/robocook/data', 'planning', 
        f'data_{tool_type}_action={max_n_actions}_p={tool_args.n_particles}')
    os.system('mkdir -p ' + plan_dataset_root)

    for dataset_name in ['train', 'valid', 'test']:
        plan_dataset_path = os.path.join(plan_dataset_root, dataset_name)
        dy_dataset_path = os.path.join(dy_dataset_root, dataset_name)

        dy_dataset_size = len(glob.glob(os.path.join(dy_dataset_path, '*')))
        print(f"Rolling out on the {dataset_name} set:")
        for idx in tqdm(range(dy_dataset_size)):
            dy_vid_path = os.path.join(dy_dataset_path, str(idx).zfill(3))
            if not os.path.exists(dy_vid_path): 
                continue

            plan_vid_path = os.path.join(plan_dataset_path, str(idx).zfill(3))
            if os.path.exists(plan_vid_path) and len(glob.glob(os.path.join(plan_vid_path, '*'))) == sample_size:
                continue

            state_init = load_data(tool_args.data_names, os.path.join(dy_vid_path, '000.h5'))[0]
            state_cur = torch.tensor(state_init[None, :tool_args.n_particles, :3], dtype=torch.float32, device=device)

            center = torch.mean(state_cur.squeeze(), dim=0).cpu()
            min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
            max_bounds = torch.max(state_cur, dim=1).values.squeeze().cpu().numpy()
            param_bounds = get_param_bounds(tool_args, tool_params, min_bounds, max_bounds)

            # param_seqs = None
            # for _ in range(max_n_actions):
            #     param_sample_list = []
            #     for i in range(param_bounds.shape[0]):
            #         # param_seq_sample = np.linspace(*param_bounds[i], num=sample_ratio[i]+1, endpoint=False)[1:]
            #         param_sample = np.random.uniform(*param_bounds[i], size=sample_ratio[i])
            #         param_sample_list.append(param_sample)

            #     param_grid = np.meshgrid(*param_sample_list)
            #     param_seq_sample = np.vstack([x.ravel() for x in param_grid]).T
            #     param_seq_sample = np.expand_dims(param_seq_sample, axis=1)

            #     if param_seqs is None:
            #         param_seqs = param_seq_sample
            #     else:
            #         np.random.shuffle(param_seq_sample)
            #         param_seqs = np.concatenate((param_seqs, param_seq_sample), axis=1)

            param_bounds = torch.tile(param_bounds, (max_n_actions, 1, 1))
            param_seqs = torch.rand(sample_size, param_bounds.shape[0], param_bounds.shape[1], dtype=torch.float32) * \
                (param_bounds[:, :, 1] - param_bounds[:, :, 0]) + param_bounds[:, :, 0]

            init_pose_seqs = []
            act_seqs = []
            for param_seq in param_seqs:
                init_pose_seq = params_to_init_pose(tool_args, center, tool_params, param_seq)
                act_seq = params_to_actions(tool_args, tool_params, param_seq, min_bounds)
                init_pose_seqs.append(init_pose_seq)
                act_seqs.append(act_seq)

            init_pose_seqs = torch.stack(init_pose_seqs)
            act_seqs = torch.stack(act_seqs)

            keyframe_list = act_seqs.shape[2] // tool_args.time_step * np.array(range(1, act_seqs.shape[1]+1)) - 1
            n_batch = int(np.ceil(param_seqs.shape[0] / B))
            state_seqs = []

            for i in range(n_batch):
                start = B * i
                end = min(B * (i + 1), param_seqs.shape[0])

                with torch.no_grad():
                    state_seq, _, _ = gnn.rollout(copy.deepcopy(state_cur), init_pose_seqs[start: end], act_seqs[start: end])

                state_seqs.append(state_seq)

            state_seqs = torch.cat(state_seqs, dim=0)
            state_kf = torch.cat((torch.tile(state_cur.unsqueeze(0), (param_seqs.shape[0], 1, 1, 1)), 
                state_seqs[:, keyframe_list]), dim=1)

            state_kf_numpy = state_kf.cpu().numpy()
            param_seq_numpy = param_seqs.cpu().numpy()
            for i in range(state_kf_numpy.shape[0]):
                plan_vid_sample_path = os.path.join(plan_vid_path, str(i).zfill(3))
                os.system('mkdir -p ' + plan_vid_sample_path)

                with open(os.path.join(plan_vid_sample_path, 'param_seq.npy'), 'wb') as f:
                    np.save(f, param_seq_numpy[i])

                for j in range(state_kf_numpy.shape[1]):
                    state_normals = get_normals(state_kf_numpy[i][j][None])[0]
                    state = np.concatenate([state_kf_numpy[i][j], state_normals], axis=1)
                    state_data = [state[:args.n_particles], shape_quats, tool_args.scene_params]
                    store_data(tool_args.data_names, state_data, os.path.join(plan_vid_sample_path, f'{str(j).zfill(3)}.h5'))
                    # gc.collect()

                if i < 3:
                    titles = ['Initial State'] + [[round(x, 3) for x in param_seq_numpy[i][j]] for j in range(param_seq_numpy.shape[1])] 
                    render_frames(tool_args, titles, state_kf_numpy[i][:, None, :, :], res='high', axis_off=True, focus=True,
                        path=plan_vid_sample_path, name=f'{str(idx).zfill(3)}_{str(i).zfill(3)}.png')

                    act_seq_sparse = params_to_actions(tool_args, tool_params, param_seqs[i], min_bounds, step=tool_args.time_step)
                    state_seq_wshape = add_shape_to_seq(tool_args, state_seqs[i].squeeze().cpu().numpy(), init_pose_seqs[i].cpu().numpy(), act_seq_sparse.cpu().numpy())
                    render_anim(tool_args, ['GNN'], [state_seq_wshape], res='low', 
                        path=os.path.join(plan_vid_sample_path, f'repr.mp4'))


def main():
    args = gen_args()

    dy_root = '/scr/hshi74/projects/robocook/dump/dynamics/dump'
    suffix = 'robot_v4_surf_nocorr_full_normal_keyframe=16'
    tool_model_dict = {
        # 'gripper_sym_rod': f'{dy_root}_gripper_sym_rod_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.005_0.005_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-19-17:27:26',
        # 'gripper_asym': f'{dy_root}_gripper_asym_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.007_0.007_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-20-23:52:40',
        # 'gripper_sym_plane': f'{dy_root}_gripper_sym_plane_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.007_0.007_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-20-23:52:48',
        # 'punch_square': f'{dy_root}_punch_square_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.004_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Oct-30-22:40:40',
        # 'punch_circle': f'{dy_root}_punch_square_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.004_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Oct-30-22:40:40',
        # 'press_square': f'{dy_root}_press_square_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.006_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Nov-02-14:11:03',
        # 'press_circle': f'{dy_root}_press_square_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.006_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Nov-02-14:11:03',
        # 'roller_large': f'{dy_root}_roller_large_{suffix}/' + \
        #     'dy_keyframe_nr=0.01_tnr=0.005_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Nov-02-19:01:19'
        'roller_small': f'{dy_root}_roller_small_{suffix}/' + \
            'dy_keyframe_nr=0.01_tnr=0.005_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Dec-22-18:34:28'
    }

    for tool_name, tool_model_path in tool_model_dict.items():
        build_planning_dataset(args, tool_name, tool_model_path)


if __name__ == '__main__':
    main()
