import glob
import os
import open3d as o3d
import numpy as np

from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *
from perception.pcd_utils import *
from perception.sample import *
from tqdm import tqdm


def process_pcd(args, bag_path, n_particles=4096, visualize=False):
    pcd_msgs = []
    while True:
        try:
            bag = rosbag.Bag(bag_path)  # allow_unindexed=True
            break
        except rosbag.bag.ROSBagUnindexedException:
            print("Reindex the rosbag file:")
            os.system(f"rosbag reindex {bag_path}")
            bag_orig_path = os.path.join(os.path.dirname(bag_path), "pcd.orig.bag")
            os.system(f"rm {bag_orig_path}")
        except rosbag.bag.ROSBagException:
            continue

    for topic, msg, t in bag.read_messages(
        topics=[
            "/cam1/depth/color/points",
            "/cam2/depth/color/points",
            "/cam3/depth/color/points",
            "/cam4/depth/color/points",
            "/ee_pose",
            "/gripper_width",
        ]
    ):
        if "cam" in topic:
            pcd_msgs.append(msg)

    bag.close()

    if "hook" in args.env or "spatula_large" in args.env or "spatula_small" in args.env:
        pcd = merge_point_cloud(
            args,
            pcd_msgs,
            crop_range=[-0.1, -0.25, 0.02, 0.1, -0.05, 0.08],
            visualize=visualize,
        )
    else:
        pcd = merge_point_cloud(
            args,
            pcd_msgs,
            crop_range=[-0.05, -0.05, 0.002, 0.1, 0.1, 0.07],
            visualize=visualize,
        )

    # cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize, rm_stats_outliers=2)
    cube = pcd

    dough_points = np.asarray(cube.points)
    dough_colors = np.asarray(cube.colors)

    if len(dough_points) > 0:
        farthest_pts = np.zeros((n_particles, 3))
        farthest_colors = np.zeros((n_particles, 3))
        first_idx = np.random.randint(len(dough_points))
        farthest_pts[0] = dough_points[first_idx]
        farthest_colors[0] = dough_colors[first_idx]
        distances = calc_distances(farthest_pts[0], dough_points)
        for i in range(1, n_particles):
            next_idx = np.argmax(distances)
            farthest_pts[i] = dough_points[next_idx]
            farthest_colors[i] = dough_colors[next_idx]
            distances = np.minimum(
                distances, calc_distances(farthest_pts[i], dough_points)
            )

        dough_points = farthest_pts
        dough_colors = farthest_colors

    dough = o3d.geometry.PointCloud()
    dough.points = o3d.utility.Vector3dVector(dough_points)
    dough.colors = o3d.utility.Vector3dVector(dough_colors)

    return dough


def build_dataset(root, target_path, augmented=1):
    args = gen_args()
    visualize = False

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
        print(f"{dataset_name} set size: {len(ep_path_list)}")
        tool_data_dict = {}
        for ep_path in ep_path_list:
            seq_path_list = sorted(glob.glob(os.path.join(ep_path, "*")))
            i = 0
            tool_name_prev = "_".join(os.path.basename(seq_path_list[0]).split("_")[2:])
            while i < len(seq_path_list):
                in_seq_path = seq_path_list[i]
                tool_name = "_".join(os.path.basename(in_seq_path).split("_")[2:])
                # if dataset_name == 'train' and tool_name in ['gripper_sym_plane', 'roller_large']:
                #     i += 1
                #     continue
                in_data_paths = sorted(glob.glob(os.path.join(in_seq_path, "in.bag")))

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
                        # out_seq_idx_list.append(len(seq_path_list) - 1)
                        # out_seq_idx_list.append(i)

                        for dumpling_ep_path in ep_data_dict["dumpling"]:
                            dumpling_seq_path_list = sorted(
                                glob.glob(os.path.join(dumpling_ep_path, "*"))
                            )
                            out_seq_path = dumpling_seq_path_list[-1]
                            out_data_paths = sorted(
                                glob.glob(os.path.join(out_seq_path, "out.bag"))
                            )
                            in_out_data_pair = in_data_paths + out_data_paths

                            if tool_name in tool_data_dict:
                                tool_data_dict[tool_name].append(in_out_data_pair)
                            else:
                                tool_data_dict[tool_name] = [in_out_data_pair]

                    tool_name_prev = tool_name

                elif augmented == 0:
                    out_seq_idx_list = [i]
                else:
                    out_seq_idx_list = list(range(i, len(seq_path_list)))

                # augment the data pairs
                for j in out_seq_idx_list:
                    out_seq_path = seq_path_list[j]
                    out_data_paths = sorted(
                        glob.glob(os.path.join(out_seq_path, "out.bag"))
                    )
                    in_out_data_pair = in_data_paths + out_data_paths

                    if tool_name in tool_data_dict:
                        tool_data_dict[tool_name].append(in_out_data_pair)
                    else:
                        tool_data_dict[tool_name] = [in_out_data_pair]

                i += 1

        for tool_name, tool_data_pair_list in tool_data_dict.items():
            # if not 'punch_square' in tool_name:
            #     continue
            args.env = tool_name
            # if args.env in args.tool_dim:
            #     shape_quats = np.zeros((sum(args.tool_dim[args.env]) + args.floor_dim, 4), dtype=np.float32)
            # else:
            #     shape_quats = np.zeros((args.floor_dim, 4), dtype=np.float32)

            print(f"========== {tool_name.upper()} ==========")
            tool_data_dir = os.path.join(target_path, dataset_name, tool_name)
            size = len(tool_data_pair_list)
            ep_idx = 0
            for i in tqdm(range(size), desc=f"{dataset_name}"):
                ep_path_out = os.path.join(tool_data_dir, str(ep_idx).zfill(3))
                if (
                    os.path.exists(ep_path_out)
                    and len(glob.glob(os.path.join(ep_path_out, "*"))) > 2
                ):
                    ep_idx += 1
                    continue

                in_src = "_".join(tool_data_pair_list[i][0].split("/")[-3:-1])
                out_src = "_".join(tool_data_pair_list[i][-1].split("/")[-3:-1])
                src_info = f"in: {in_src}\nout: {out_src}"

                has_error = False
                os.system("mkdir -p " + ep_path_out)
                if len(tool_data_pair_list[i]) != 2:
                    has_error = True
                    print(f"{src_info}")
                else:
                    state_name_list = ["in", "out"]
                    # state_pair = []
                    for j in range(2):
                        pcd_path = tool_data_pair_list[i][j]
                        try:
                            cube = process_pcd(args, pcd_path, visualize=False)
                        except RuntimeError:
                            has_error = True
                            break

                        # state_normals = get_normals(state_points[None])[0]

                        # if visualize:
                        #     dough_pcd = o3d.geometry.PointCloud()
                        #     dough_pcd.points = o3d.utility.Vector3dVector(state_points)
                        #     dough_pcd.normals = o3d.utility.Vector3dVector(state_normals)
                        #     o3d.visualization.draw_geometries([dough_pcd], point_show_normal=True)

                        # state = np.concatenate([state_points, state_normals], axis=1)
                        # state_pair.append(state)
                        # h5_data = [state, shape_quats, args.scene_params]
                        # store_data(args.data_names, h5_data, os.path.join(ep_path_out, state_name_list[j] + '.h5'))
                        o3d.io.write_point_cloud(
                            os.path.join(ep_path_out, state_name_list[j] + ".ply"), cube
                        )

                if has_error:
                    os.system(f"rm -r {ep_path_out}")
                else:
                    # state_pair = np.stack(state_pair)
                    with open(os.path.join(ep_path_out, "src.txt"), "w") as f:
                        f.write(src_info)

                    # render_frames(args, [f'Before {tool_name.upper()}', f'After {tool_name.upper()}'],
                    #     [state_pair[0:1, :args.n_particles], state_pair[1:2, :args.n_particles]],
                    #     res='low', axis_off=False, path=ep_path_out, name="pair.png")

                    ep_idx += 1

                # break


def main():
    root = "/media/hshi74/Game Drive PS4/robocook/raw_data/classifier_08-24"
    target_path = "data/pcd_classifier/classifier_08-24_v4"

    build_dataset(root, target_path, augmented=1)


if __name__ == "__main__":
    main()
