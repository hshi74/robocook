from matplotlib.pyplot import title
import open3d as o3d

import cv2 as cv
import copy
import glob
import numpy as np
import os
import rosbag
import ros_numpy
import sys
import trimesh

from datetime import datetime
from perception.pcd_utils import *
from pysdf import SDF
from timeit import default_timer as timer
from transforms3d.quaternions import *
from tqdm import tqdm
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *


# @profile
def merge_point_cloud(
    args, pcd_msgs, crop_range=[-0.1, -0.1, 0.002, 0.1, 0.1, 0.07], visualize=False
):
    pcd_all_list = []
    for i in range(len(pcd_msgs)):
        cloud_rec = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msgs[i])
        cloud_array = cloud_rec.view("<f4").reshape(cloud_rec.shape + (-1,))
        points = cloud_array[:, :3]
        points = (
            quat2mat(args.depth_optical_frame_pose[3:]) @ points.T
        ).T + args.depth_optical_frame_pose[:3]
        cam_ori = args.cam_pose_dict[f"cam_{i+1}"]["orientation"]
        points = (quat2mat(cam_ori) @ points.T).T + args.cam_pose_dict[f"cam_{i+1}"][
            "position"
        ]

        cloud_rgb_bytes = cloud_array[:, -1].tobytes()
        # int.from_bytes(cloud_rgb_bytes, 'big')
        cloud_bgr = np.frombuffer(cloud_rgb_bytes, dtype=np.uint8).reshape(-1, 4) / 255
        cloud_rgb = cloud_bgr[:, ::-1]

        if crop_range is None:
            x_filter = (points.T[0] > args.mid_point[0] - 0.5) & (
                points.T[0] < args.mid_point[0] + 0.3
            )
            y_filter = (points.T[1] > -0.45) & (points.T[1] < 0.45)
            z_filter = (points.T[2] > args.mid_point[2] - 0.02) & (
                points.T[2] < args.mid_point[2] + 1.0
            )
        else:
            x_filter = (points.T[0] > args.mid_point[0] + crop_range[0]) & (
                points.T[0] < args.mid_point[0] + crop_range[3]
            )
            y_filter = (points.T[1] > args.mid_point[1] + crop_range[1]) & (
                points.T[1] < args.mid_point[1] + crop_range[4]
            )
            z_filter = (points.T[2] > args.mid_point[2] + crop_range[2]) & (
                points.T[2] < args.mid_point[2] + crop_range[5]
            )

        points = points[x_filter & y_filter & z_filter]
        cloud_rgb = cloud_rgb[x_filter & y_filter & z_filter, 1:]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(cloud_rgb)

        pcd_all_list.append(pcd)

    pcd_all = o3d.geometry.PointCloud()
    for point_id in range(len(pcd_all_list)):
        pcd_all += pcd_all_list[point_id]

    # cl, inlier_ind = pcd_all.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    # pcd_all = pcd_all.select_by_index(inlier_ind)

    # pcd_all.voxel_down_sample(voxel_size=voxel_size)

    if visualize:
        if crop_range is None:
            view_point = {
                "front": [0.6, 0.0, 0.8],
                "lookat": [0.4, -0.1, 0.0],
                "up": [-0.8, 0.0, 0.6],
                "zoom": 0.5,
            }
            # cl, inlier_ind = pcd_all.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
            # pcd_all = pcd_all.select_by_index(inlier_ind)

            visualize_o3d(
                [pcd_all], view_point=view_point, title="merged_raw_point_cloud"
            )
        else:
            # cl, inlier_ind = pcd_all.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
            # pcd_all = pcd_all.select_by_index(inlier_ind)
            visualize_o3d([pcd_all], title="merged_and_cropped_raw_point_cloud")

    # cloud_path = os.path.join(dataset_path, f'{vid_idx:03d}', f'pcd_{j:03d}.ply')
    # o3d.io.write_point_cloud(cloud_path, pcd_all)

    return pcd_all


# @profile
def preprocess_raw_pcd(args, pcd_all, rm_stats_outliers=2, visualize=False):
    # segment_models, inliers = raw_pcd.segment_plane(distance_threshold=0.003,ransac_n=3,num_iterations=100)
    # pcd_all = raw_pcd.select_by_index(inliers, invert=True)

    # if visualize:
    #     visualize_o3d([pcd_all], title='segmented_pcd')

    # robocraft: lower += 0.005
    # lower = np.array([args.mid_point[0] - 0.1, args.mid_point[1] - 0.1, args.mid_point[2] + 0.005])
    # upper = np.array([args.mid_point[0] + 0.1, args.mid_point[1] + 0.1, args.mid_point[2] + 0.05])
    # pcd_all = pcd_all.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))

    pcd_colors = np.asarray(pcd_all.colors, dtype=np.float32)
    # bgr
    pcd_rgb = pcd_colors[None, :, :]

    pcd_hsv = cv.cvtColor(pcd_rgb, cv.COLOR_RGB2HSV)
    hsv_lower = np.array([0.0, 0.0, 0.0])
    hsv_upper = np.array([80.0, 255.0, 255.0])
    mask = cv.inRange(pcd_hsv, hsv_lower, hsv_upper)
    cube_label = np.where(mask[0] == 255)

    # cube_label = np.where((pcd_colors[:, 0] > 0.7) & (pcd_colors[:, 1] > 0.7) & (pcd_colors[:, 2] > 0.7))
    cube = pcd_all.select_by_index(cube_label[0])
    # rest_label = np.where(np.logical_and(pcd_colors[:, 0] > 0.5, pcd_colors[:, 2] > 0.2))
    # rest = pcd_all.select_by_index(rest_label[0])
    rest = pcd_all.select_by_index(cube_label[0], invert=True)

    if visualize:
        visualize_o3d([cube], title="selected_dough")

    if visualize:
        visualize_o3d([rest], title="discarded_part")

    cube = cube.voxel_down_sample(voxel_size=0.001)

    if rm_stats_outliers:
        rm_iter = 0
        outliers = None
        outlier_stat = None
        # remove until there's no new outlier
        while outlier_stat is None or len(outlier_stat.points) > 0:
            cl, inlier_ind_cube_stat = cube.remove_statistical_outlier(
                nb_neighbors=50, std_ratio=1.5 + 0.5 * rm_iter
            )
            cube_stat = cube.select_by_index(inlier_ind_cube_stat)
            outlier_stat = cube.select_by_index(inlier_ind_cube_stat, invert=True)
            if outliers is None:
                outliers = outlier_stat
            else:
                outliers += outlier_stat

            # print(len(outlier.points))

            # cl, inlier_ind_cube_stat = cube.remove_radius_outlier(nb_points=50, radius=0.05)
            # cube_stat = cube.select_by_index(inlier_ind_cube_stat)
            # outliers = cube.select_by_index(inlier_ind_cube_stat, invert=True)

            cube = cube_stat
            rm_iter += 1

            # press needs those points
            if "press" in args.env or rm_stats_outliers == 1:
                break

        # cl, inlier_ind_rest_stat = rest.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
        # rest_stat = rest.select_by_index(inlier_ind_rest_stat)
        # rest = rest_stat

        if visualize:
            outliers.paint_uniform_color([0.0, 0.8, 0.0])
            visualize_o3d([cube, outliers], title="cleaned_dough")

    # rest = rest.voxel_down_sample(voxel_size=0.004)

    # if visualize:
    #     visualize_o3d([cube, rest], title='downsampled_dough_and_tool')

    # n_bins = 30
    # cube_points = np.asarray(cube.points)
    # cube_colors = np.asarray(cube.colors)
    # cube_z_hist_array = np.histogram(cube_points[:, 2], bins=n_bins)
    # # cube_z_count_max = np.argmax(cube_z_hist_array[0])
    # cube_z_cap = cube_z_hist_array[1][n_bins - 1]
    # selected_idx = cube_points[:, 2] < cube_z_cap
    # cube_points = cube_points[selected_idx]
    # cube_colors = cube_colors[selected_idx]

    # cube = o3d.geometry.PointCloud()
    # cube.points = o3d.utility.Vector3dVector(cube_points)
    # cube.colors = o3d.utility.Vector3dVector(cube_colors)

    # if visualize:
    #     visualize_o3d([cube], title='cube_under_ceiling')

    return cube, rest


def inside_tool_filter(
    sampled_points, tool_list, in_d=0.0, close_d=-0.01, visualize=False
):
    sdf_all = np.full(sampled_points.shape[0], True)

    n_points_close = 0
    for tool_mesh, _ in tool_list:
        f = SDF(tool_mesh.vertices, tool_mesh.triangles)
        sdf = f(sampled_points)
        sdf_all &= sdf < in_d
        n_points_close += np.sum(sdf > close_d)

    out_tool_points = sampled_points[sdf_all, :]
    in_tool_points = sampled_points[~sdf_all, :]

    if visualize:
        out_tool_pcd = o3d.geometry.PointCloud()
        out_tool_pcd.points = o3d.utility.Vector3dVector(out_tool_points)
        out_tool_pcd.paint_uniform_color([0.6, 0.6, 0.6])

        in_tool_pcd = o3d.geometry.PointCloud()
        in_tool_pcd.points = o3d.utility.Vector3dVector(in_tool_points)
        in_tool_pcd.paint_uniform_color([0.0, 0.0, 0.0])

        visualize_o3d(
            [*list(zip(*tool_list))[1], out_tool_pcd, in_tool_pcd],
            title="inside_tool_filter",
        )

    return out_tool_points, in_tool_points


def check_if_close(cube, tool_list, close_d=-0.01):
    cube_hull, _ = cube.compute_convex_hull()
    f = SDF(cube_hull.vertices, cube_hull.triangles)

    # a sparse pass
    for _, tool_surface in tool_list:
        tool_surface_sparse = tool_surface.voxel_down_sample(voxel_size=0.01)
        sdf = f(np.asarray(tool_surface_sparse.points))
        n_points_close = np.sum(sdf > close_d)
        if n_points_close > 0:
            return True

    return False


# @profile
def inside_cube_filter(cube, tool_list, visualize=False):
    hull, _ = cube.compute_convex_hull()
    f = SDF(hull.vertices, hull.triangles)

    patch_list = []
    for _, tool_surface in tool_list:
        surface_points = np.asarray(tool_surface.points)

        sdf = f(surface_points)
        surface_points = surface_points[sdf > 0]

        patch_list.append(surface_points)

    patch_points = np.concatenate(patch_list, axis=0)
    patch_pcd = o3d.geometry.PointCloud()
    patch_pcd.points = o3d.utility.Vector3dVector(patch_points)

    cube_patched = patch_pcd + cube
    # cube_patched = cube_patched.voxel_down_sample(voxel_size=0.004)

    if visualize:
        visualize_o3d([cube_patched], title="inside_cube_filter")

    return cube_patched


# @profile
def sample(
    args,
    pcd,
    pcd_dense_prev,
    pcd_sparse_prev,
    tool_list,
    is_moving_back,
    patch=False,
    visualize=False,
):
    if pcd_dense_prev is not None and is_moving_back:
        return pcd_dense_prev, pcd_sparse_prev

    cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize)
    is_close = check_if_close(cube, tool_list)

    if pcd_dense_prev is not None and not is_close:
        ##### 5.a apply temporal prior: copy the soln from the last frame #####
        return pcd_dense_prev, pcd_sparse_prev

    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    ##### 1. random sample 100x points in the bounding box #####
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    sample_size = 50 * args.n_particles
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    # import pdb; pdb.set_trace()
    # visualize = True

    if patch and is_close:
        cube = inside_cube_filter(cube, tool_list, visualize=visualize)

    # #### 2.a use SDF to filter out points OUTSIDE the convex hull #####

    # f = SDF(convex_hull.vertices, convex_hull.triangles)
    # sdf = f(sampled_points)
    # sampled_points = sampled_points[sdf > 0, :]

    # cube = cube.voxel_down_sample(voxel_size=0.004)

    if "roller" in args.env or "press" in args.env:
        selected_mesh = alpha_shape_mesh_reconstruct(
            cube, alpha=0.2, mesh_fix=False, visualize=visualize
        )
        f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    else:
        selected_mesh = poisson_mesh_reconstruct(
            cube, depth=6, mesh_fix=True, visualize=visualize
        )
        f = SDF(
            selected_mesh.points,
            selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:],
        )

    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]

    if not "gripper" in args.env:
        vg_mask = vg_filter(pcd, sampled_points, visualize=visualize)
        sampled_points = sampled_points[vg_mask]

    ##### 3. use SDF to filter out points INSIDE the tool mesh #####
    sampled_points, _ = inside_tool_filter(
        sampled_points, tool_list, visualize=visualize
    )
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

    if visualize:
        visualize_o3d(
            [sampled_pcd, cube, *list(zip(*tool_list))[1]], title="sampled_points"
        )

    ##### 6. filter out the noise #####
    if not "roller" in args.env or not "press" in args.env:
        cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=1.5
        )
        sampled_pcd_stat = sampled_pcd.select_by_index(inlier_ind_stat)
        outliers_stat = sampled_pcd.select_by_index(inlier_ind_stat, invert=True)

        # cl, inlier_ind_radi = sampled_pcd_stat.remove_radius_outlier(nb_points=300, radius=0.01)
        # sampled_pcd_radi = sampled_pcd_stat.select_by_index(inlier_ind_radi)
        # outliers_radi = sampled_pcd_stat.select_by_index(inlier_ind_radi, invert=True)

        # sampled_pcd = sampled_pcd_radi
        # outliers = outliers_stat + outliers_radi

        sampled_pcd = sampled_pcd_stat
        outliers = outliers_stat

        if visualize:
            sampled_pcd.paint_uniform_color([0.0, 0.8, 0.0])
            outliers.paint_uniform_color([0.8, 0.0, 0.0])
            visualize_o3d(
                [cube, sampled_pcd, outliers],
                title="cleaned_point_cloud",
                pcd_color=color_avg,
            )

    ##### (optional) 8. surface sampling #####
    if args.surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(
            sampled_pcd, alpha=0.005, visualize=visualize
        )

        if not args.correspondance or pcd_dense_prev is None:
            selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                selected_mesh, args.n_particles
            )
            surface_points = np.asarray(selected_surface.points)
        else:
            tri_mesh = trimesh.Trimesh(
                np.asarray(selected_mesh.vertices),
                np.asarray(selected_mesh.triangles),
                vertex_normals=np.asarray(selected_mesh.vertex_normals),
            )
            mesh_q = trimesh.proximity.ProximityQuery(tri_mesh)
            prox_points, distance, triangle_id = mesh_q.on_surface(
                np.asarray(pcd_sparse_prev.points)
            )
            selector = (distance > 0.0)[..., None]
            surface_points = prox_points * selector + np.asarray(
                pcd_sparse_prev.points
            ) * (1 - selector)

        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        if visualize:
            visualize_o3d(
                [surface_pcd], title="surface_point_cloud", pcd_color=color_avg
            )

        selected_pcd = surface_pcd
    else:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles)
        fps_pcd = o3d.geometry.PointCloud()
        fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        if visualize:
            visualize_o3d(
                [fps_pcd, *list(zip(*tool_list))[1]],
                title="fps_point_cloud",
                pcd_color=color_avg,
            )

        selected_pcd = fps_pcd

    return sampled_pcd, selected_pcd


# @profile
def ros_bag_to_pcd(
    args,
    bag_path,
    tool_list,
    pcd_dense_prev=None,
    pcd_sparse_prev=None,
    image_path_prefix="",
    last_dist=float("inf"),
    is_moving_back=False,
    visualize=False,
    write=False,
):
    pcd_msgs = []
    try:
        bag = rosbag.Bag(bag_path)
    except rosbag.bag.ROSBagUnindexedException:
        os.system(f"rosbag reindex {bag_path}")
        bag = rosbag.Bag(bag_path)

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

        if topic == "/ee_pose":
            ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
            ee_quat = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )

        if topic == "/gripper_width":
            gripper_width = msg.data

    bag.close()

    pcd = merge_point_cloud(args, pcd_msgs, visualize=visualize)

    # transform the tool mesh
    fingertip_mat = args.ee_fingertip_T_mat[:3, :3] @ quat2mat(ee_quat)

    # The data for asym gripping is collected incorrectly
    if "asym" in args.env:
        fingertip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ fingertip_mat

    fingermid_pos = (quat2mat(ee_quat) @ args.ee_fingertip_T_mat[:3, 3].T).T + ee_pos

    tool_name_list = args.tool_geom_mapping[args.env]
    tool_list_T = []
    fingertip_T_list = []
    for k in range(len(tool_name_list)):
        if "gripper" in args.env:
            fingertip_pos = (
                fingertip_mat @ np.array([(1 - 2 * k) * (gripper_width) / 2, 0, 0]).T
            ).T + fingermid_pos
        else:
            fingertip_pos = fingermid_pos
        fingertip_T = np.concatenate(
            (
                np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1),
                [[0, 0, 0, 1]],
            ),
            axis=0,
        )
        fingertip_T_list.append(fingertip_T)

        tool_mesh_T = copy.deepcopy(tool_list[k][0]).transform(fingertip_T)
        tool_surface_T = copy.deepcopy(tool_list[k][1]).transform(fingertip_T)
        tool_list_T.append((tool_mesh_T, tool_surface_T))

    if visualize:
        visualize_o3d([pcd, *list(zip(*tool_list_T))[1]], title="transformed_tool_mesh")

    if write:
        visualize_o3d(
            [pcd, *list(zip(*tool_list_T))[1]],
            title="transformed_tool_mesh",
            path=os.path.join(image_path_prefix + "_raw.png"),
        )

    if "gripper" in args.env:
        dist = gripper_width / 2
        # leveraging motion prior
        if not is_moving_back and dist > last_dist + 0.001:
            is_moving_back = True
            print("Start moving back...")
        if is_moving_back and dist < last_dist - 0.001:
            is_moving_back = False
            print("End moving back...")
    else:
        dist = np.linalg.norm(args.mid_point - fingertip_T_list[0][:3, 3])

    pcd_dense, pcd_sparse = sample(
        args,
        pcd,
        pcd_dense_prev,
        pcd_sparse_prev,
        tool_list_T,
        is_moving_back,
        patch=False,
        visualize=visualize,
    )

    tool_repr = get_tool_repr(args, fingertip_T_list)
    state_cur = np.concatenate(
        [np.asarray(pcd_sparse.points), args.floor_state, tool_repr]
    )

    return pcd_dense, pcd_sparse, state_cur, dist, is_moving_back


def main():
    args = gen_args()

    tool_name_list = args.tool_geom_mapping[args.env]
    tool_list = []
    for i in range(len(tool_name_list)):
        tool_mesh = o3d.io.read_triangle_mesh(
            os.path.join(args.tool_geom_path, f"{tool_name_list[i]}.stl")
        )
        tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
            tool_mesh, 10000
        )
        tool_list.append((tool_mesh, tool_surface))

    write_frames = False
    write_gt_state = False
    visualize = False

    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    rollout_dir = os.path.join(
        cd, "..", "dump", "perception", f"{args.tool_type}_{time_now}"
    )

    data_root = os.path.join("data", "raw", args.tool_type)

    dir_list = sorted(glob.glob(os.path.join(data_root, "*")))
    episode_len = 5 if "gripper" in args.env else 3

    start_idx = input("Please enter the start index (int):\n")  # 0
    n_vids = input("Please enter the range (int):\n")  # len(dir_list)
    for i in range(int(start_idx), int(start_idx) + int(n_vids)):  # len(dir_list)
        # vid_idx = os.path.basename(dir_list[i])
        vid_idx = str(i).zfill(3)
        # print(f'========== Video {vid_idx} ==========')

        bag_path = os.path.join(
            data_root,
            f"ep_{str(i // episode_len).zfill(3)}",
            f"seq_{str(i % episode_len).zfill(3)}",
        )

        bag_list = sorted(
            glob.glob(os.path.join(bag_path, "*.bag")),
            key=lambda x: float(os.path.basename(x)[:-4]),
        )

        if len(bag_list) == 0:
            print(f"Video {vid_idx} does not exist!")
            break

        rollout_path = os.path.join(rollout_dir, vid_idx)
        image_path = os.path.join(rollout_path, "images")
        os.system("mkdir -p " + rollout_path)
        if write_frames:
            os.system("mkdir -p " + f"{rollout_path}/frames")
        if write_gt_state:
            os.system("mkdir -p " + f"{image_path}")

        state_seq = []
        pcd_dense_prev, pcd_sparse_prev = None, None
        is_moving_back = False
        last_dist = float("inf")
        start_frame = 0
        step_size = 1
        for j in tqdm(
            range(start_frame, len(bag_list), step_size), desc=f"Video {vid_idx}"
        ):  # len(bag_list)
            # ros_bag_debug_path = "misc/state_0.bag"
            image_path_prefix = os.path.join(image_path, str(j).zfill(3))
            (
                pcd_dense,
                pcd_sparse,
                state_cur,
                last_dist,
                is_moving_back,
            ) = ros_bag_to_pcd(
                args,
                bag_list[j],
                tool_list,
                pcd_dense_prev=pcd_dense_prev,
                pcd_sparse_prev=pcd_sparse_prev,
                image_path_prefix=image_path_prefix,
                last_dist=last_dist,
                is_moving_back=is_moving_back,
                visualize=visualize,
                write=write_gt_state,
            )

            pcd_dense_prev = pcd_dense
            pcd_sparse_prev = pcd_sparse

            # if state_prev is not None:
            # state_rebuild = np.concatenate((state_prev[:args.n_particles], state_cur[args.n_particles:]))
            state_seq.append(state_cur)

            shape_quats = np.zeros(
                (sum(args.tool_dim[args.env]) + args.floor_dim, 4), dtype=np.float32
            )
            h5_data = [state_cur, shape_quats, args.scene_params]
            store_data(
                args.data_names,
                h5_data,
                os.path.join(rollout_path, str(j).zfill(3) + ".h5"),
            )

            # state_prev = state_cur

        render_anim(
            args,
            ["Perception"],
            [np.array(state_seq)],
            path=os.path.join(rollout_path, "repr.mp4"),
        )
        if write_frames:
            render_frames(
                args,
                ["Perception"],
                [np.array(state_seq)],
                views=[(90, 0)],
                path=os.path.join(rollout_path, "frames"),
            )
        if write_gt_state:
            os.system(
                f"ffmpeg -f image2 -framerate 20 -start_number {start_frame} -i {image_path}/%03d_raw.png {rollout_path}/raw.mp4"
            )
            os.system(f"rm -r {image_path}")


if __name__ == "__main__":
    main()
