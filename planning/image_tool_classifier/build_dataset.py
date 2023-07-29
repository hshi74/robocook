import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm


def crop_helper(im, cam_idx, image_path):
    if "1" in cam_idx:
        if (
            "hook" in image_path
            or "spatula_large" in image_path
            or ("spatula_small" in image_path and "out" in image_path)
        ):
            crop_range = (350, 250, 650, 550)
        else:
            crop_range = (650, 150, 850, 350)
    elif "2" in cam_idx:
        if (
            "hook" in image_path
            or "spatula_large" in image_path
            or ("spatula_small" in image_path and "out" in image_path)
        ):
            crop_range = (300, 200, 500, 400)
        else:
            crop_range = (400, 325, 700, 475)
    elif "3" in cam_idx:
        if (
            "hook" in image_path
            or "spatula_large" in image_path
            or ("spatula_small" in image_path and "out" in image_path)
        ):
            crop_range = (600, 150, 800, 350)
        else:
            crop_range = (500, 300, 750, 500)
    else:
        if (
            "hook" in image_path
            or "spatula_large" in image_path
            or ("spatula_small" in image_path and "out" in image_path)
        ):
            crop_range = (650, 150, 950, 450)
        else:
            crop_range = (400, 150, 700, 400)

    im_coarse = im.crop(crop_range)
    im_array = np.array(im_coarse)

    im_hsv = cv.cvtColor(im_array, cv.COLOR_RGB2HSV)
    hsv_lower = np.array([0, 0, 150])
    hsv_upper = np.array([60, 255, 255])
    mask = cv.inRange(im_hsv, hsv_lower, hsv_upper)

    return crop_range, im_coarse, mask


def crop_image(image_path, dim=200, visualize=False):
    im = Image.open(image_path)

    # crop the center of the image
    cam_idx = os.path.basename(image_path).split("_")[-1]
    crop_range, im_coarse, mask = crop_helper(im, cam_idx, image_path)
    x, y = np.nonzero(mask)
    if x.shape[0] < 1000:
        crop_range, im_coarse, mask = crop_helper(im, cam_idx, image_path + "hook")
        x, y = np.nonzero(mask)

    x_mean = int(np.mean(x))
    y_mean = int(np.mean(y))

    # if visualize:
    #     _, axs = plt.subplots(1, 3, figsize=(8, 24))
    #     axs = axs.flatten()
    #     for img, ax in zip([im, im_coarse, mask], axs):
    #         ax.imshow(img)
    #     plt.show()

    # k-means clustersing
    # Z = np.vstack(np.nonzero(mask)).astype(np.float32).T
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # A = Z[label.ravel() == 0]
    # B = Z[label.ravel() == 1]

    # if A.shape[0] > B.shape[0]:
    #     x_mean, y_mean = int(center[0][0]), int(center[0][1])
    # else:
    #     x_mean, y_mean = int(center[1][0]), int(center[1][1])

    left = crop_range[0] + y_mean - dim / 2
    top = crop_range[1] + x_mean - dim / 2
    right = crop_range[0] + y_mean + dim / 2
    bottom = crop_range[1] + x_mean + dim / 2
    im_fine = im.crop((left, top, right, bottom))

    if visualize:
        _, axs = plt.subplots(2, 2, figsize=(16, 16))
        axs = axs.flatten()
        for img, ax in zip([im, im_coarse, mask, im_fine], axs):
            ax.imshow(img)
        plt.show()

    # import pdb; pdb.set_trace()

    return im_fine


def build_dataset(root, target_path, augmented=True):
    # organize all the data into a dictionary
    tool_data_dict = {}
    ep_path_list = sorted(glob.glob(os.path.join(root, "*")))
    for ep_path in ep_path_list:
        seq_path_list = sorted(glob.glob(os.path.join(ep_path, "*")))
        for i in range(len(seq_path_list)):
            in_seq_path = seq_path_list[i]
            tool_name = "_".join(os.path.basename(in_seq_path).split("_")[2:])
            in_data_paths = sorted(glob.glob(os.path.join(in_seq_path, "in*.png")))

            if augmented:
                out_seq_idx_end = len(seq_path_list)
            else:
                out_seq_idx_end = i + 1
            # augment the data pairs
            for j in range(i, out_seq_idx_end):
                out_seq_path = seq_path_list[j]
                out_data_paths = sorted(
                    glob.glob(os.path.join(out_seq_path, "out*.png"))
                )
                in_out_data_pair = in_data_paths + out_data_paths

                if tool_name in tool_data_dict:
                    tool_data_dict[tool_name].append(in_out_data_pair)
                else:
                    tool_data_dict[tool_name] = [in_out_data_pair]

    # split into train, valid, test
    for tool_name, tool_data_pair_list in tool_data_dict.items():
        # if 'gripper'in tool_name:
        #     continue

        # if not 'cutter' in tool_name:
        #     continue

        dataset_size = len(tool_data_pair_list)
        idx_list = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(idx_list)

        valid_set_size = int(dataset_size * 0.1)
        test_set_size = int(dataset_size * 0.1)
        training_set_size = dataset_size - valid_set_size - test_set_size

        print(f"========== {tool_name.upper()} ==========")
        print(f"Training set size: {training_set_size}")
        print(f"Valid set size: {valid_set_size}")
        print(f"Test set size: {test_set_size}")

        dataset_dict = {
            "train": training_set_size,
            "valid": valid_set_size,
            "test": test_set_size,
        }
        data_idx = 0
        for dataset, size in dataset_dict.items():
            tool_data_dir = os.path.join(target_path, dataset, tool_name)
            for i in tqdm(range(size), desc=f"{dataset}"):
                in_src = "_".join(
                    tool_data_pair_list[idx_list[data_idx]][0].split("/")[-3:-1]
                )
                out_src = "_".join(
                    tool_data_pair_list[idx_list[data_idx]][-1].split("/")[-3:-1]
                )
                src_info = f"in: {in_src}\nout: {out_src}"
                final_im_dict = {}
                if len(tool_data_pair_list[idx_list[data_idx]]) != 8:
                    has_error = True
                else:
                    has_error = False
                    for j in range(8):
                        image_path = tool_data_pair_list[idx_list[data_idx]][j]
                        # print(image_path)
                        try:
                            final_im = crop_image(image_path)
                            final_im_dict[image_path] = final_im
                        except ValueError:
                            has_error = True
                            # crop_image(image_path, visualize=True)
                            break

                if has_error:
                    i -= 1
                    size -= 1
                    print(f"{src_info}")
                else:
                    ep_path = os.path.join(tool_data_dir, str(i).zfill(3))
                    os.system("mkdir -p " + ep_path)
                    with open(os.path.join(ep_path, "src.txt"), "w") as f:
                        f.write(src_info)

                    for image_path, final_im in final_im_dict.items():
                        final_im.save(
                            os.path.join(ep_path, os.path.basename(image_path))
                        )

                data_idx += 1


def build_dataset_epwise(root, target_path, augmented=2):
    # organize all the data into a dictionary
    ep_data_dict = {}
    ep_path_list = sorted(glob.glob(os.path.join(root, "*")))
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

    dataset_dict = {"train": [], "valid": [], "test": []}
    for tool_name, ep_path_list in ep_data_dict.items():
        if tool_name in [
            "gripper_asym",
            "gripper_sym_rod",
            "press_circle",
            "punch_circle",
            "punch_square",
            "roller_small",
        ]:
            continue
        dataset_size = len(ep_path_list)
        print(f"{tool_name}: {dataset_size}")

        idx_list = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(idx_list)

        valid_set_size = int(dataset_size * 0.1)
        test_set_size = int(dataset_size * 0.1)
        training_set_size = dataset_size - valid_set_size - test_set_size

        data_idx = 0
        while data_idx < dataset_size:
            if data_idx < training_set_size:
                dataset_dict["train"].append(ep_path_list[idx_list[data_idx]])
            elif data_idx < training_set_size + valid_set_size:
                dataset_dict["valid"].append(ep_path_list[idx_list[data_idx]])
            else:
                dataset_dict["test"].append(ep_path_list[idx_list[data_idx]])

            data_idx += 1

    for dataset_name, ep_path_list in dataset_dict.items():
        # if 'train' in dataset_name:
        #     continue
        print(f"{dataset_name} set size: {len(ep_path_list)}")
        tool_data_dict = {}
        for ep_path in ep_path_list:
            seq_path_list = sorted(glob.glob(os.path.join(ep_path, "*")))
            i = 0
            tool_name_prev = "_".join(os.path.basename(seq_path_list[0]).split("_")[2:])
            while i < len(seq_path_list):
                in_seq_path = seq_path_list[i]
                tool_name = "_".join(os.path.basename(in_seq_path).split("_")[2:])
                if dataset_name == "train" and tool_name in [
                    "gripper_sym_plane",
                    "roller_large",
                ]:
                    i += 1
                    continue
                in_data_paths = sorted(glob.glob(os.path.join(in_seq_path, "in*.png")))

                if augmented == 1 and len(seq_path_list) > 10:
                    out_seq_idx_list = []
                    if tool_name in [
                        "gripper_sym_plane",
                        "roller_large",
                        "press_square",
                    ]:
                        j = i
                        tool_name_next = tool_name
                        while (
                            tool_name_next == tool_name and j < len(seq_path_list) - 1
                        ):
                            out_seq_idx_list.append(j)
                            j += 1
                            tool_name_next = "_".join(
                                os.path.basename(seq_path_list[j]).split("_")[2:]
                            )

                    if i == 0 or tool_name != tool_name_prev:
                        out_seq_idx_list.append(len(seq_path_list) - 1)
                        # for dumpling_ep_path in ep_data_dict['dumpling']:
                        #     dumpling_seq_path_list = sorted(glob.glob(os.path.join(dumpling_ep_path, '*')))
                        #     out_seq_path = dumpling_seq_path_list[-1]
                        #     out_data_paths = sorted(glob.glob(os.path.join(out_seq_path, 'out*.png')))
                        #     in_out_data_pair = in_data_paths + out_data_paths

                        #     if tool_name in tool_data_dict:
                        #         tool_data_dict[tool_name].append(in_out_data_pair)
                        #     else:
                        #         tool_data_dict[tool_name] = [in_out_data_pair]

                    tool_name_prev = tool_name

                elif augmented == 0:
                    out_seq_idx_list = [i]
                else:
                    out_seq_idx_list = list(range(i, len(seq_path_list)))

                # augment the data pairs
                for j in out_seq_idx_list:
                    out_seq_path = seq_path_list[j]
                    out_data_paths = sorted(
                        glob.glob(os.path.join(out_seq_path, "out*.png"))
                    )
                    in_out_data_pair = in_data_paths + out_data_paths

                    if tool_name in tool_data_dict:
                        tool_data_dict[tool_name].append(in_out_data_pair)
                    else:
                        tool_data_dict[tool_name] = [in_out_data_pair]

                i += 1

        for tool_name, tool_data_pair_list in tool_data_dict.items():
            print(f"========== {tool_name.upper()} ==========")
            tool_data_dir = os.path.join(target_path, dataset_name, tool_name)
            size = len(tool_data_pair_list)
            for i in tqdm(range(size), desc=f"{dataset_name}"):
                in_src = "_".join(tool_data_pair_list[i][0].split("/")[-3:-1])
                out_src = "_".join(tool_data_pair_list[i][-1].split("/")[-3:-1])
                src_info = f"in: {in_src}\nout: {out_src}"
                final_im_dict = {}
                if len(tool_data_pair_list[i]) != 8:
                    has_error = True
                else:
                    has_error = False
                    for j in range(8):
                        image_path = tool_data_pair_list[i][j]
                        # print(image_path)
                        try:
                            final_im = crop_image(image_path)
                            final_im_dict[image_path] = final_im
                        except ValueError:
                            has_error = True
                            # crop_image(image_path, visualize=True)
                            break

                if has_error:
                    i -= 1
                    size -= 1
                    print(f"{src_info}")
                else:
                    ep_path_out = os.path.join(tool_data_dir, str(i).zfill(3))
                    os.system("mkdir -p " + ep_path_out)
                    with open(os.path.join(ep_path_out, "src.txt"), "w") as f:
                        f.write(src_info)

                    for image_path, final_im in final_im_dict.items():
                        final_im.save(
                            os.path.join(ep_path_out, os.path.basename(image_path))
                        )


def main():
    root = "data/raw_data/classifier_08-24"
    target_path = "data/image_classifier/classifier_08-24_epwise_final_v2.5"

    # build_dataset(root, target_path)
    build_dataset_epwise(root, target_path, augmented=1)


if __name__ == "__main__":
    main()
