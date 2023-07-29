import glob
import open3d as o3d
import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_utils import *


class ToolDataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(args.tool_dataf, phase)
        self.data_names = ["positions", "shape_quats", "scene_params"]

        self.classes = sorted([d.name for d in os.scandir(self.data_dir) if d.is_dir()])
        self.class_to_idx = {class_name: x for x, class_name in enumerate(self.classes)}

        print(f"Loading {phase} data...")
        self.point_set = self._get_data()

    def _get_data(self):
        point_set = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[target_class]
            seq_path_list = sorted(
                glob.glob(os.path.join(self.data_dir, target_class, "*"))
            )
            for seq_path in tqdm(seq_path_list, desc=f"{target_class}"):
                sample_path_list = sorted(glob.glob(os.path.join(seq_path, "*.ply")))
                if len(sample_path_list) < 2:
                    continue
                state_list = []
                for k in range(len(sample_path_list)):
                    state_pcd = o3d.io.read_point_cloud(sample_path_list[k])
                    state = np.concatenate(
                        (np.asarray(state_pcd.points), np.asarray(state_pcd.colors)),
                        axis=1,
                    )
                    if not self.args.use_rgb:
                        state = state[:, :3]
                    if self.args.early_fusion:
                        state = np.concatenate(
                            (state, np.full((state.shape[0], 1), k)), axis=1
                        )
                    state_list.append(state)
                state_concat = np.concatenate(state_list, axis=0)

                centroid = np.mean(state_list[0][:, :3], axis=0)
                state_concat[:, :3] = state_concat[:, :3] - centroid
                m = np.max(np.sqrt(np.sum(state_concat[:, :3] ** 2, axis=1)))
                state_concat[:, :3] = state_concat[:, :3] / m

                point_set.append((state_concat, class_idx))

        return point_set

    def __len__(self):
        return len(self.point_set)

    def __getitem__(self, index):
        return self.point_set[index]
