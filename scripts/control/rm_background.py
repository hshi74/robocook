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


def remove_bg(img_path, img_nobg_path, visualize=False):
    im = Image.open(img_path)
    im_array = np.array(im)

    im_hsv = cv.cvtColor(im_array, cv.COLOR_RGB2HSV)
    hsv_lower = np.array([0, 0, 0])
    hsv_upper = np.array([40, 255, 255])
    mask = cv.inRange(im_hsv, hsv_lower, hsv_upper)

    # im_fine = np.random.randint(low=0, high=255, size=im_array.shape, dtype=np.uint8)
    im_fine = np.zeros_like(im_array, dtype=np.uint8)
    im_fine[mask > 0] = im_array[mask > 0]

    if visualize:
        _, axs = plt.subplots(1, 3, figsize=(24, 8))
        axs = axs.flatten()
        for img, ax in zip([im, mask, im_fine], axs):
            ax.imshow(img)
        plt.show()

    im_new = Image.fromarray(im_fine)
    im_new.save(img_nobg_path)


def make_dataset():
    root = "data/image_classifier/classifier_08-24_epwise_final_v3_nobg"
    visualize = False
    for dataset in ["train", "valid", "test"]:
        tool_data_path_list = sorted(glob.glob(os.path.join(root, dataset, "*")))
        for tool_data_path in tool_data_path_list:
            ep_path_list = sorted(glob.glob(os.path.join(tool_data_path, "*")))
            for ep_path in tqdm(ep_path_list, desc=os.path.basename(tool_data_path)):
                img_path_list = sorted(glob.glob(os.path.join(ep_path, "*.png")))
                for img_path in img_path_list:
                    remove_bg(img_path, img_path, visualize=visualize)


def update_target():
    root = "target_shapes/dumpling"
    visualize = False
    target_path_list = sorted(glob.glob(os.path.join(root, "*")))
    for target_path in target_path_list:
        img_path_list = sorted(glob.glob(os.path.join(target_path, "*cam*.png")))
        for img_path in img_path_list:
            img_new_name = os.path.basename(img_path).split(".")[0] + "_nobg.png"
            img_new_path = os.path.join(os.path.dirname(img_path), img_new_name)
            if "nobg" in img_path:
                os.system(f"rm {img_path}")
            else:
                remove_bg(img_path, img_new_path, visualize=visualize)


def main():
    # make_dataset()
    update_target()


if __name__ == "__main__":
    main()
