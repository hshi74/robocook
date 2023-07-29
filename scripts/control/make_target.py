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


target_root = "/scr/hshi74/projects/robocook/target_shapes/alphabet/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_target(
    args, letter_name, step, state_init, tool_name, tool_model_path, params
):
    print(letter_name)
    tool_type = tool_model_path.split("/")[-2].replace("dump_", "")
    dy_dataset_root = os.path.join(
        "/scr/hshi74/projects/robocook/data", "keyframe", f"data_{tool_type}"
    )

    tool_args = copy.deepcopy(args)
    tool_args.env = tool_name

    # dy_args_dict = np.load(f'{tool_model_path}_args.npy', allow_pickle=True).item()
    dy_args_dict = np.load(f"{tool_model_path}/args.npy", allow_pickle=True).item()
    tool_args = update_dy_args(tool_args, tool_name, dy_args_dict)

    with open("/scr/hshi74/projects/robocook/config/tool_plan_params.yml", "r") as f:
        tool_params = yaml.load(f, Loader=yaml.FullLoader)[tool_args.env]

    shape_quats = np.zeros(
        (sum(tool_args.tool_dim[tool_args.env]) + tool_args.floor_dim, 4),
        dtype=np.float32,
    )

    # print_args(tool_args)
    # gnn = GNN(tool_args, f'{tool_model_path}.pth')
    gnn = GNN(tool_args, f"{tool_model_path}/net_best.pth")

    if not tool_name in tool_type:
        tool_type = tool_type.replace("square", "circle")

    letter_root = os.path.join(
        target_root, letter_name, f"{str(step).zfill(3)}_{tool_name}"
    )
    os.system("mkdir -p " + letter_root)

    state_cur = torch.tensor(
        state_init[None, : tool_args.n_particles, :3],
        dtype=torch.float32,
        device=device,
    )

    center = torch.mean(state_cur.squeeze(), dim=0).cpu()
    min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()

    param_seq = torch.tensor(params, dtype=torch.float32)

    init_pose_seq = params_to_init_pose(tool_args, center, tool_params, param_seq)
    act_seq = params_to_actions(tool_args, tool_params, param_seq, min_bounds)

    with torch.no_grad():
        state_seq, _, _ = gnn.rollout(
            copy.deepcopy(state_cur), init_pose_seq.unsqueeze(0), act_seq.unsqueeze(0)
        )

    state_final_numpy = state_seq[0][-1].cpu().numpy()
    param_seq_numpy = param_seq.cpu().numpy()

    with open(os.path.join(letter_root, "param_seq.npy"), "wb") as f:
        np.save(f, param_seq_numpy)

    state_data = [
        state_final_numpy[: args.n_particles],
        shape_quats,
        tool_args.scene_params,
    ]
    store_data(
        tool_args.data_names,
        state_data,
        os.path.join(letter_root, f"{str(step).zfill(3)}_surf.h5"),
    )

    os.system(
        f"cp -r {letter_root.replace('gnn', 'real')}/{str(step).zfill(3)}_raw.ply "
        + f"{letter_root}/{str(step).zfill(3)}_raw.ply"
    )

    titles = ["Initial State"] + [[round(x, 3) for x in param_seq_numpy[-1]]]
    render_frames(
        tool_args,
        titles,
        [state_init[None], state_final_numpy[None]],
        res="low",
        axis_off=False,
        focus=False,
        path=letter_root,
        name=f"{str(step).zfill(3)}_sampled.png",
    )

    act_seq_sparse = params_to_actions(
        tool_args, tool_params, param_seq, min_bounds, step=tool_args.time_step
    )
    state_seq_wshape = add_shape_to_seq(
        tool_args,
        state_seq.squeeze().cpu().numpy(),
        init_pose_seq.cpu().numpy(),
        act_seq_sparse.cpu().numpy(),
    )
    render_anim(
        tool_args,
        ["GNN"],
        [state_seq_wshape],
        res="low",
        path=os.path.join(letter_root, f"{str(step).zfill(3)}_anim.mp4"),
    )


def main():
    args = gen_args()

    dy_root = "/scr/hshi74/projects/robocook/dump/dynamics/dump"
    suffix = "robot_v4_surf_nocorr_full_normal_keyframe=16"
    tool_model_dict = {
        "gripper_sym_rod": f"{dy_root}_gripper_sym_rod_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.005_0.005_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-19-17:27:26",
        "gripper_asym": f"{dy_root}_gripper_asym_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.007_0.007_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-20-23:52:40",
        "gripper_sym_plane": f"{dy_root}_gripper_sym_plane_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.007_0.007_his=1_seq=2_time_step=3_chamfer_emd_0.5_0.5_rm=1_valid_Oct-20-23:52:48",
        "punch_square": f"{dy_root}_punch_square_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.004_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Oct-30-22:40:40",
        "punch_circle": f"{dy_root}_punch_square_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.004_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Oct-30-22:40:40",
        "press_square": f"{dy_root}_press_square_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.006_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Nov-02-14:11:03",
        "roller_large": f"{dy_root}_roller_large_{suffix}/"
        + "dy_keyframe_nr=0.01_tnr=0.005_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Nov-02-19:01:19",
    }

    letters_dict = {
        "R": {
            "tool": ["gripper_sym_rod", "punch_circle"],
            "params": [[[0.017, -0.15, 0.0]], [[-0.0075, 0.0, 0.005]]],
        },
        # 'R': {
        #     'tool': ['gripper_asym', 'punch_circle'],
        #     'params': [[[0.01,1.9, 0.01], [-0.01, 2.355, 0.005]], [[-0.0075, 0.0, 0.005]]]
        # },
        # 'O': {
        #     'tool': ['gripper_sym_plane', 'punch_circle'],
        #     'params': [[[0.0, 0.0, 0.036], [0.0, 1.57, 0.036]], [[0.0, 0.0, 0.005]]]
        # },
        # 'B': {
        #     'tool': ['gripper_asym', 'punch_circle'],
        #     'params': [[[0.0, 2.355, 0.01]], [[0.016, 0.0, 0.005], [-0.016, 0.0, 0.005]]]
        # },
        # 'C': {
        #     'tool': ['gripper_asym'],
        #     'params': [[[0.005, 2.1, 0.005], [-0.01, 2.6, 0.005]]]
        # },
        # 'K': {
        #     'tool': ['gripper_sym_rod', 'gripper_asym'],
        #     'params': [[[0.0, 0.785, 0.01]], [[0.0, 2.355, 0.005]]]
        # }
    }

    for letter_name, info in letters_dict.items():
        # os.system(f'cp -r /scr/hshi74/projects/robocook/target_shapes/alphabet/{letter_name}/000 ' + \
        #     f'/scr/hshi74/projects/robocook/target_shapes/alphabet/{letter_name}/000')

        for i in range(len(info["tool"])):
            if i == 0:
                state_path = os.path.join(
                    target_root.replace("gnn", "real"),
                    letter_name,
                    f"{str(i).zfill(3)}/{str(i).zfill(3)}.h5",
                )
            else:
                state_path = os.path.join(
                    target_root.replace("gnn", "real"),
                    letter_name,
                    f"{str(i).zfill(3)}_{info['tool'][i-1]}/{str(i).zfill(3)}.h5",
                )

            state_cur = load_data(args.data_names, state_path)[0]
            make_target(
                args,
                letter_name,
                i + 1,
                state_cur,
                info["tool"][i],
                tool_model_dict[info["tool"][i]],
                info["params"][i],
            )


if __name__ == "__main__":
    main()
