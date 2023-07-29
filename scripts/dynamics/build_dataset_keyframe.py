import glob
import numpy as np
import os

from tqdm import tqdm
from utils.config import *
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *

def main():
    args = gen_args()

    key_frame=16
    keyframe_dataset_root = os.path.join('data', 'keyframe', f'data_{args.tool_type}_keyframe={key_frame}')
    os.system('mkdir -p ' + keyframe_dataset_root)

    for dataset_name in ['train', 'valid', 'test']:
        key_frame_dataset_path = os.path.join(keyframe_dataset_root, dataset_name)
        dy_dataset_path = os.path.join(args.dy_data_path, dataset_name)

        dy_dataset_size = len(glob.glob(os.path.join(dy_dataset_path, '*')))
        print(f"Rolling out on the {dataset_name} set:")
        for idx in tqdm(range(dy_dataset_size)):
            keyframe_vid_path = os.path.join(key_frame_dataset_path, str(idx).zfill(3))
            if os.path.exists(keyframe_vid_path) and len(glob.glob(os.path.join(keyframe_vid_path, '*'))) > 1:
                continue

            os.system('mkdir -p ' + keyframe_vid_path)
            dy_vid_path = os.path.join(dy_dataset_path, str(idx).zfill(3))
            if not os.path.exists(dy_vid_path): 
                continue

            # load data
            state_seq = []
            frame_list = sorted(glob.glob(os.path.join(dy_vid_path, '*.h5')))
            last_grip_width = float('inf')
            for step in range(len(frame_list)):
                frame_name = str(step).zfill(3) + '.h5'
                state = load_data(args.data_names, os.path.join(dy_vid_path, frame_name))[0]
                tool_center_list = []
                tool_start = args.n_particles + args.floor_dim
                for i in range(len(args.tool_dim[args.env])):
                    tool_dim = args.tool_dim[args.env][i]
                    tool_center_list.append(np.mean(state[tool_start:tool_start+tool_dim, :3], axis=0))
                    tool_start += tool_dim
                if 'gripper' in args.env:
                    grip_width = np.linalg.norm(tool_center_list[1] - tool_center_list[0])
                    if grip_width > last_grip_width + 0.002:
                        print(f"Start moving back at {step}...")
                        break
                    else:
                        state_seq.append(state)
                    last_grip_width = grip_width
                else:
                    state_seq.append(state)

            state_seq = np.stack(state_seq)

            if 'roller' in args.env:
                keyframe_idx_list = [int(x) for x in np.round(np.arange(0, state_seq.shape[0], (state_seq.shape[0] - 1) / (key_frame - 1)))]
            else:
                action_seq = state_seq[1:, args.n_particles + args.floor_dim, :3] - state_seq[:-1, args.n_particles + args.floor_dim, :3]
                action_seq_cum = np.cumsum(action_seq, axis=0)
                action_seq_cum_key_frame = np.cumsum(np.tile(action_seq_cum[-1] / (key_frame - 1), (key_frame - 1, 1)), axis=0)

                keyframe_idx_list = []
                for i in range(action_seq_cum_key_frame.shape[0] - 1):
                    dist_min = float('inf')
                    dist_min_idx = 0
                    for j in range(action_seq_cum.shape[0] - 1):
                        dist = np.linalg.norm(action_seq_cum_key_frame[i] - action_seq_cum[j])
                        if dist < dist_min:
                            dist_min_idx = j
                            dist_min = dist

                    keyframe_idx_list.append(dist_min_idx)
                keyframe_idx_list = [0] + keyframe_idx_list + [state_seq.shape[0] - 1]

            for i in range(len(keyframe_idx_list)):
                os.system(f"cp {os.path.join(dy_vid_path, str(keyframe_idx_list[i]).zfill(3) + '.h5')} " + \
                    f"{os.path.join(keyframe_vid_path, str(i).zfill(3) + '.h5')}")

            render_anim(args, ['GT'], [state_seq[keyframe_idx_list]], res='low', path=os.path.join(keyframe_vid_path, f'repr.mp4'))

            if args.debug: return


if __name__ == '__main__':
    main()
