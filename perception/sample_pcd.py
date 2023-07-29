import open3d as o3d

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
from perception.sample import *
from pysdf import SDF
from timeit import default_timer as timer
from transforms3d.quaternions import *
from tqdm import tqdm
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *


# @profile
def sample(args, pcd, use_vg_filter=False, visualize=False):
    cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize, rm_stats_outliers=2)

    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    ##### 1. random sample 100x points in the bounding box #####
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    sample_size = 50 * args.n_particles
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    # TODO: Verify if this is a good choice
    # z_extent = (cube.get_max_bound() - cube.get_min_bound())[2]
    # if z_extent < 0.02:
    # selected_mesh = alpha_shape_mesh_reconstruct(cube, alpha=0.1, mesh_fix=False, visualize=visualize)
    # f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    # else:
    selected_mesh = poisson_mesh_reconstruct(
        cube, depth=6, mesh_fix=True, visualize=visualize
    )
    f = SDF(
        selected_mesh.points,
        selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:],
    )

    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]

    if use_vg_filter:
        vg_mask = vg_filter(cube, sampled_points, visualize=visualize)
        sampled_points = sampled_points[vg_mask]

    if visualize:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        visualize_o3d([sampled_pcd, cube], title="sampled_points")

    ##### 3. use SDF to filter out points INSIDE the tool mesh #####
    # sampled_points, n_points_close = inside_tool_filter(sampled_points, tool_mesh_list, visualize=visualize)
    # print(f'points touching: {n_points_touching}')
    # print(f'is_moving_back: {is_moving_back}')

    ##### 6. filter out the noise #####
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    # sampled_pcd = sampled_pcd.voxel_down_sample(voxel_size=0.002)

    cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(
        nb_neighbors=50, std_ratio=2.0
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

    # if visualize:
    #     sampled_pcd.paint_uniform_color([0.0, 0.8, 0.0])
    #     outliers.paint_uniform_color([0.8, 0.0, 0.0])
    #     visualize_o3d([cube, sampled_pcd, outliers], title='cleaned_point_cloud', pcd_color=color_avg)

    ##### (optional) 8. surface sampling #####
    if args.surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(
            sampled_pcd, alpha=0.005, visualize=visualize
        )

        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
            selected_mesh, args.n_particles
        )
        surface_points = np.asarray(selected_surface.points)

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
            visualize_o3d([fps_pcd], title="fps_point_cloud", pcd_color=color_avg)

        selected_pcd = fps_pcd

    return sampled_pcd, selected_pcd


# @profile
def ros_bag_to_pcd(args, bag_path, use_vg_filter=False, visualize=False):
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

    ee_pos, ee_quat, gripper_width = None, None, None
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

    has_robot_info = (
        ee_pos is not None and ee_quat is not None and gripper_width is not None
    )

    if (
        "hook" in args.env or "spatula_large" in args.env or "spatula_small" in args.env
    ):  # 'out.bag' in bag_path
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

    pcd_dense, pcd_sparse = sample(
        args, pcd, use_vg_filter=use_vg_filter, visualize=visualize
    )

    state_cur = np.asarray(pcd_sparse.points)
    if has_robot_info:
        # transform the tool mesh
        fingertip_mat = args.ee_fingertip_T_mat[:3, :3] @ quat2mat(ee_quat)
        fingermid_pos = (
            quat2mat(ee_quat) @ args.ee_fingertip_T_mat[:3, 3].T
        ).T + ee_pos

        tool_name_list = args.tool_geom_mapping[args.env]
        fingertip_T_list = []
        for k in range(len(tool_name_list)):
            if "gripper" in args.env:
                fingertip_pos = (
                    fingertip_mat
                    @ np.array([(1 - 2 * k) * (gripper_width) / 2, 0, 0]).T
                ).T + fingermid_pos
            else:
                fingertip_pos = fingermid_pos
            fingertip_T = np.concatenate(
                (
                    np.concatenate(
                        (fingertip_mat, np.array([fingertip_pos]).T), axis=1
                    ),
                    [[0, 0, 0, 1]],
                ),
                axis=0,
            )
            fingertip_T_list.append(fingertip_T)

        tool_repr = get_tool_repr(args, fingertip_T_list)
        state_cur = np.concatenate(
            [np.asarray(pcd_sparse.points), args.floor_state, tool_repr]
        )

    return pcd_dense, pcd_sparse, state_cur


def main():
    args = gen_args()

    write_frames = True
    write_gt_state = True

    ros_bag_debug_path = "dump/control/control_gripper_sym_robot_v2.1_surf_nocorr/alphabet/A/control_close_max=5_CEM_40_0.5_chamfer_Jul-18-19:20:36/raw_data/state_2.bag"
    rollout_path = os.path.dirname(ros_bag_debug_path)
    _, _, state_cur = ros_bag_to_pcd(args, ros_bag_debug_path, visualize=True)

    if write_frames:
        render_frames(
            args,
            ["Perception"],
            [np.array([state_cur])],
            views=[(90, 0)],
            path=os.path.join(rollout_path),
        )


if __name__ == "__main__":
    main()
