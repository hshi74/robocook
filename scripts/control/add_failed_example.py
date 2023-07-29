import glob
import numpy as np
import open3d as o3d
import os
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm
from planning.image_tool_classifier.build_dataset import crop_image
from PIL import Image
from utils.data_utils import *
from utils.visualize import *


def crop(img_dir, new_img_dir, dim=200):
    im = Image.open(img_dir)
    width, height = im.size
    if width > dim or height > dim:
        im = crop_image(img_dir, dim=dim)
    im.save(new_img_dir)


def add_training_example():
    data_root = "/media/hshi74/Game Drive PS4/robocook/raw_data/classifier_08-24"
    target_root = "data/image_classifier/data_classifier_08-24_epwise_final_v3"
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
        + "control_close_max=5_CEM_40_0.5_chamfer_Oct-12-20:02:27/001/raw_data"
    )
    prefix = "Oct-12-20:06:03"
    label = "spatula_small"
    in_img_path_list = sorted(
        glob.glob(os.path.join(control_root, f"{prefix}_cam*.png"))
    )
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
            for in_img_path in in_img_path_list:
                img_new_name = "in" + os.path.basename(in_img_path).split(prefix)[-1]
                in_img_new_path = os.path.join(
                    target_root, dataset, label, str(idx_new).zfill(3), img_new_name
                )
                crop(in_img_path, in_img_new_path)

            dumpling_seq_path_list = sorted(
                glob.glob(os.path.join(dumpling_ep_path, "*"))
            )
            out_seq_path = dumpling_seq_path_list[-1]
            out_data_paths = sorted(glob.glob(os.path.join(out_seq_path, "out*.png")))
            for out_img_path in out_data_paths:
                out_img_new_path = os.path.join(
                    target_root,
                    dataset,
                    label,
                    str(idx_new).zfill(3),
                    os.path.basename(out_img_path),
                )
                crop(out_img_path, out_img_new_path)

            idx_new += 1


def main():
    add_training_example()


if __name__ == "__main__":
    main()
