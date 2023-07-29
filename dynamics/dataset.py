import glob
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from utils.data_utils import *


class GNNDataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(args.dy_data_path, phase)
        self.stat_path = os.path.join(args.dy_data_path, "..", "stats.h5")
        self.dataset_len = 0

        vid_path_list = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        if phase == "train":
            vid_path_list = vid_path_list[
                : int(args.train_set_ratio * len(vid_path_list))
            ]

        self.state_data_list = []
        self.action_data_list = []
        n_frames_min = float("inf")
        for vid_path in vid_path_list:
            if "synthetic" in args.data_type:
                gt_vid_path = vid_path.replace("synthetic", "gt").replace(
                    f"_time_step={args.data_time_step}", ""
                )
            else:
                gt_vid_path = vid_path

            frame_start = 0
            n_frames = len(glob.glob(os.path.join(gt_vid_path, "*.h5")))
            n_frames_min = min(n_frames, n_frames_min)
            gt_state_list = []
            gt_action_list = []
            for i in range(n_frames):
                gt_frame_data = load_data(
                    args.data_names, os.path.join(gt_vid_path, f"{str(i).zfill(3)}.h5")
                )[0]
                gt_state = torch.tensor(
                    gt_frame_data, device=args.device, dtype=torch.float32
                )
                gt_state_list.append(gt_state)
                gt_action_list.append(gt_state[args.n_particles + args.floor_dim :])

            self.action_data_list.append(torch.stack(gt_action_list))

            state_seq_list = []
            for i in range(
                frame_start,
                n_frames
                - args.time_step
                * args.data_time_step
                * (args.n_his + args.sequence_length - 1),
            ):
                state_seq = []
                # history frames
                for j in range(
                    i, i + args.time_step * (args.n_his - 1) + 1, args.time_step
                ):
                    # print(f'history: {j}')
                    state_seq.append(gt_state_list[j])

                # frames to predict
                for j in range(
                    i + args.time_step * args.n_his,
                    i + args.time_step * (args.n_his + args.sequence_length - 1) + 1,
                    args.time_step,
                ):
                    # print(f'predict: {j}')
                    if "synthetic" in args.data_type:
                        pred_frame_data = load_data(
                            args.data_names,
                            os.path.join(
                                vid_path,
                                str(i + args.time_step * (args.n_his - 1)).zfill(3),
                                f"{str(j).zfill(3)}.h5",
                            ),
                        )[0]
                        pred_state = torch.tensor(
                            pred_frame_data, device=args.device, dtype=torch.float32
                        )
                        state_seq.append(pred_state)
                    else:
                        state_seq.append(gt_state_list[j])

                self.dataset_len += 1
                state_seq_list.append(torch.stack(state_seq))

            self.state_data_list.append(state_seq_list)

        print(f"{phase} -> number of sequences: {self.dataset_len}")
        print(f"{phase} -> minimum number of frames: {n_frames_min}")

    def __len__(self):
        # Each data point consists of a sequence
        return self.dataset_len

    def load_stats(self):
        # print("Loading stat from %s ..." % self.stat_path)
        self.stat = load_data(self.args.data_names[:1], self.stat_path)

    @profile
    def __getitem__(self, idx):
        args = self.args

        idx_curr = idx
        idx_vid = 0
        offset = len(self.state_data_list[0])
        while idx_curr >= offset:
            idx_curr -= offset
            idx_vid = (idx_vid + 1) % len(self.state_data_list)
            offset = len(self.state_data_list[idx_vid])

        state_seq = self.state_data_list[idx_vid][idx_curr][
            : args.n_his + args.sequence_length
        ]
        action_seq = self.action_data_list[idx_vid][
            idx_curr : idx_curr
            + args.data_time_step
            * args.time_step
            * (args.n_his + args.sequence_length - 1)
            + 1 : args.data_time_step
        ]

        return state_seq, action_seq
