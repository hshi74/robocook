import glob
import numpy as np
import open3d as o3d
import os
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from PIL import Image
from tqdm import tqdm
from planning.pcd_tool_classifier.build_dataset import process_pcd
from PIL import Image
from utils.config import *
from utils.data_utils import *
from utils.visualize import *


def add_training_example():
    args = gen_args()

    data_root = "/media/hshi74/Game Drive PS4/robocook/raw_data/classifier_08-24"
    target_root = "data/pcd_classifier/classifier_08-24_v4"
    ep_data_dict = {}
    ep_path_list = sorted(glob.glob(os.path.join(data_root, "*")))
    for ep_path in ep_path_list:
        seq_path_list = sorted(glob.glob(os.path.join(ep_path, "*")))
        if len(seq_path_list) > 10:
            tool_name = "dumpling"
        else:
            tool_name = "_".join(os.path.basename(seq_path_list[0]).split("_")[2:])

        if tool_name in ep_data_dict:
            ep_data_dict[tool_name].append(ep_path)
        else:
            ep_data_dict[tool_name] = [ep_path]

    control_root = (
        "dump/control/control_gripper_sym_rod_robot_v4_surf_nocorr_full/dumpling/"
        + "control_close_max=2_RS_chamfer_Nov-22-22:45:09/001/raw_data"
    )
    prefix = "Nov-22-22:48:26"
    label = "press_square"
    args.env = label
    in_pcd_path = os.path.join(control_root, f"{prefix}.bag")
    dataset_range = {"train": (0, 8), "valid": (8, 9), "test": (9, 10)}
    for dataset in ["train", "valid", "test"]:
        data_list = sorted(glob.glob(os.path.join(target_root, dataset, label, f"*")))
        idx_new = len(data_list)
        for dumpling_ep_path in ep_data_dict["dumpling"][
            dataset_range[dataset][0] : dataset_range[dataset][1]
        ]:
            os.system(
                f"mkdir -p {os.path.join(target_root, dataset, label, str(idx_new).zfill(3))}"
            )
            in_pcd_new_path = os.path.join(
                target_root, dataset, label, str(idx_new).zfill(3), "in.ply"
            )
            in_pcd = process_pcd(args, in_pcd_path)
            o3d.io.write_point_cloud(in_pcd_new_path, in_pcd)

            dumpling_seq_path_list = sorted(
                glob.glob(os.path.join(dumpling_ep_path, "*"))
            )
            out_seq_path = dumpling_seq_path_list[-1]
            out_pcd_path = os.path.join(out_seq_path, "out.bag")
            out_pcd_new_path = os.path.join(
                target_root, dataset, label, str(idx_new).zfill(3), "out.ply"
            )
            out_pcd = process_pcd(args, out_pcd_path)
            o3d.io.write_point_cloud(out_pcd_new_path, out_pcd)

            idx_new += 1


def main():
    add_training_example()


if __name__ == "__main__":
    main()
