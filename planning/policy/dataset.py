import glob
import os
import numpy as np

from perception.pcd_utils import *
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_utils import *


class PolicyDataset(Dataset):
    def __init__(self, args, phase, rollout_ratio=1.0):
        self.args = args
        self.phase = phase
        self.rollout_ratio = rollout_ratio
        self.data_dir = os.path.join(args.plan_dataf, phase)
        self.data_names = ["positions", "shape_quats", "scene_params"]

        print(f"Loading {phase} data...")
        self.point_set = self._get_data()

    def _get_data(self):
        seq_path_list = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        state_concat_list = []
        param_seq_list = []
        for i in range(len(seq_path_list)):
            sample_path_list = sorted(glob.glob(os.path.join(seq_path_list[i], "*")))
            for j in range(len(sample_path_list)):
                if j >= self.rollout_ratio * 128:
                    break

                state_path_list = sorted(
                    glob.glob(os.path.join(sample_path_list[j], "*.h5"))
                )
                if len(state_path_list) < self.args.n_actions + 1:
                    continue

                param_seq_data = np.load(
                    os.path.join(sample_path_list[j], "param_seq.npy")
                )
                for k in range(len(state_path_list) - self.args.n_actions):
                    state_in = load_data(self.data_names, state_path_list[k])[0]
                    state_out = load_data(
                        self.data_names, state_path_list[k + self.args.n_actions]
                    )[0]

                    state_in_center = np.mean(state_in[:, :3], axis=0)
                    state_in[:, :3] = state_in[:, :3] - state_in_center
                    state_in[:, :3] = state_in[:, :3] / self.args.scale_ratio

                    state_out_center = np.mean(state_out[:, :3], axis=0)
                    state_out[:, :3] = state_out[:, :3] - state_out_center
                    state_out[:, :3] = state_out[:, :3] / self.args.scale_ratio

                    if not self.args.use_normals:
                        state_in = state_in[:, :3]
                        state_out = state_out[:, :3]
                    if self.args.early_fusion:
                        state_in = np.concatenate(
                            (state_in, np.full((state_in.shape[0], 1), 0)), axis=1
                        )
                        state_out = np.concatenate(
                            (state_out, np.full((state_out.shape[0], 1), 1)), axis=1
                        )
                    state_concat = np.concatenate((state_in, state_out), axis=0)
                    state_concat_list.append(state_concat)

                    params = param_seq_data[k : k + self.args.n_actions].flatten()
                    for ii in range(params.shape[0]):
                        if (
                            ii % (params.shape[0] // self.args.n_actions)
                            != self.args.rot_idx
                        ):
                            params[ii] /= self.args.scale_ratio
                    param_seq_list.append(params)

        param_seq = np.stack(param_seq_list).astype(np.float32)
        param_seq_norm = (
            param_seq  # (param_seq - self.args.stats['mean']) / self.args.stats['std']
        )

        bin_ind_list = []
        for i in range(self.args.n_bins.shape[0]):
            if (
                i % (self.args.n_bins.shape[0] // self.args.n_actions)
                == self.args.rot_idx
            ):
                period_ratio = np.pi * 2 / self.args.tool_params["rot_scope"]
                bin_diff = -np.cos(
                    (
                        np.tile(param_seq_norm[:, i : i + 1], (1, self.args.n_bins[i]))
                        - self.args.bin_centers[i]
                    )
                    * period_ratio
                )
            else:
                bin_diff = np.abs(
                    np.tile(param_seq_norm[:, i : i + 1], (1, self.args.n_bins[i]))
                    - self.args.bin_centers[i]
                )
            bin_idx_list = [
                np.argsort(bin_diff[j])[:2] for j in range(param_seq_norm.shape[0])
            ]
            bin_ind_list.append(np.stack(bin_idx_list))

        bin_ind = np.stack(bin_ind_list).transpose((1, 0, 2))

        # for i in range(self.args.n_bins.shape[0]):
        #     print(np.bincount(bin_ind[:, i].astype(np.int64)))

        point_set = []
        for state_concat, bin_idx, params_norm in zip(
            state_concat_list, bin_ind, param_seq_norm
        ):
            point_set.append((state_concat, bin_idx, params_norm))

        return point_set

    def __len__(self):
        return len(self.point_set)

    def __getitem__(self, index):
        return self.point_set[index]
