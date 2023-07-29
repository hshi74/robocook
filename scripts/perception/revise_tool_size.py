import glob
import numpy as np
import os
import sys

from tqdm import tqdm
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *

def revise_tool_repr(args, tool_repr, tool_dim_before, tool_dim_after):
    if 'gripper' in args.env:
        unit_size = 0.05 / (tool_dim_after[0] - 1)
        
        def get_gripper_repr(i, gripper_center):
            tool_repr_new = []
            for j in range(tool_dim_after[i]):
                p = [gripper_center[args.axes[0]], gripper_center[args.axes[1]]]
                p.insert(args.axes[2], gripper_center[args.axes[2]] + 
                    unit_size * (j - (tool_dim_after[i] - 1) / 2))
                tool_repr_new.append(p)

            return tool_repr_new

        gripper_center_l = (tool_repr[0] + tool_repr[tool_dim_before[0] - 1]) / 2
        tool_repr_l = get_gripper_repr(0, gripper_center_l)

        if tool_dim_after[0] == tool_dim_after[1]:
            gripper_center_r = (tool_repr[tool_dim_before[0]] + tool_repr[-1]) / 2
            tool_repr_r = get_gripper_repr(1, gripper_center_r)
        else:
            tool_repr_r = np.concatenate([tool_repr[tool_dim_before[0]:][x*tool_dim_after[0]:(x+1)*tool_dim_after[0]] 
                for x in range(0, int(tool_dim_before[1]/tool_dim_after[0]), 2)], axis=0) 

        tool_repr_new = np.concatenate((tool_repr_l, tool_repr_r))
    elif 'roller' in args.env:
        unit_size = 0.07 / (tool_dim_after[0] - 1)
        tool_repr_new = []
        roller_center = (tool_repr[0] + tool_repr[tool_dim_before[0] - 1]) / 2
        for j in range(tool_dim_after[0]):
            p = [roller_center[args.axes[1]], roller_center[args.axes[2]]]
            p.insert(args.axes[0], roller_center[args.axes[0]] + 
                unit_size * (j - (tool_dim_after[0] - 1) / 2))
            tool_repr_new.append(p)
    elif 'press' in args.env or 'punch' in args.env:
        press_center = np.mean(tool_repr, axis=0)
        if 'large' in args.env:
            unit_size = 0.04 / (int(np.sqrt(tool_dim_after[0])) - 1)
        else:
            raise NotImplementedError
        tool_repr_new = get_square(press_center, unit_size, tool_dim_after[0], 2)
    else:
        raise NotImplementedError

    return tool_repr_new


def main():
    args = gen_args()

    tool_type = "gripper_asym_robot_v1.6_copy"
    tool_type_new = "gripper_asym_robot_v1.61"

    debug = True

    tool_dim_before = [11, 143]
    tool_dim_after= [11, 77]

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", "gt")
    src_dir = os.path.join(data_root_dir, f"data_{tool_type}")
    
    dataset_list = ["train", "valid", "test"]
    for dataset in dataset_list:
        dp_dir_list = sorted(glob.glob(os.path.join(src_dir, dataset, '*')))
        for i in range(len(dp_dir_list)):
            if debug and i > 0: return

            dest_dir = os.path.join(data_root_dir, f'data_{tool_type_new}', dataset, str(i).zfill(3))
            os.system(f"mkdir -p {dest_dir}")

            dp = dp_dir_list[i]
            frame_list = sorted(glob.glob(os.path.join(dp, '*.h5')))
            state_seq = []
            for j in tqdm(range(len(frame_list))):
                frame = frame_list[j]
                frame_data = load_data(args.data_names, frame)
                tool_repr_new = revise_tool_repr(args, frame_data[0][args.n_particles+args.floor_dim:], tool_dim_before, tool_dim_after)
                state_new = np.concatenate((frame_data[0][:args.n_particles+args.floor_dim], tool_repr_new), axis=0)
                state_seq.append(state_new)
                h5_data = [state_new, frame_data[1], frame_data[2]]
                store_data(args.data_names, h5_data, os.path.join(dest_dir, str(j).zfill(3) + '.h5'))

            render_anim(args, ['Perception'], [np.array(state_seq)], path=os.path.join(dest_dir, 'repr.mp4'))


if __name__ == "__main__":
    main()
