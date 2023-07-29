import copy
import glob
import numpy as np
import os
import sys

from perception.pcd_utils import *
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *


def get_rot_mat(A, B):
    # https://math.stackexchange.com/a/897677
    def ssc(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    R = np.eye(3) + ssc(np.cross(A, B)) + np.linalg.matrix_power(ssc(np.cross(A, B)), 2) * \
        (1 - np.dot(A, B)) / (np.linalg.norm(np.cross(A,B)) ** 2)

    return R


def get_finger_T_list(args, tool_repr):
    tool_dim_list = args.tool_dim[args.env]
    tool_center_list = args.tool_center[args.env]

    fingertip_T_list = []
    if 'gripper' in args.env:
        unit_vec = [1, 0, 0]
        tool_repr_list = [tool_repr[:tool_dim_list[0]], tool_repr[tool_dim_list[0]:]]
        gripper_vec = np.mean(tool_repr_list[0], axis=0) - np.mean(tool_repr_list[1], axis=0)
        gripper_vec /= np.linalg.norm(gripper_vec)
        fingertip_mat = get_rot_mat(unit_vec, gripper_vec)
        
        for tool_repr_part, tool_center in zip(tool_repr_list, tool_center_list):
            fingertip_pos = np.mean(tool_repr_part, axis=0) - fingertip_mat @ tool_center
            fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
            fingertip_T_list.append(fingertip_T)
    elif 'roller' in args.env:
        unit_vec = [1, 0, 0]
        roller_vec = tool_repr[-1] - tool_repr[0]
        roller_vec /= np.linalg.norm(roller_vec)
        fingertip_mat = get_rot_mat(unit_vec, roller_vec)
        fingertip_pos = np.mean(tool_repr, axis=0) - fingertip_mat @ tool_center_list[0]
        fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
        fingertip_T_list.append(fingertip_T)
    elif 'press' in args.env or 'punch' in args.env:
        unit_vec = [1, 0, 0]
        plane_vec = tool_repr[1] - tool_repr[0]
        plane_vec /= np.linalg.norm(plane_vec)
        fingertip_mat = get_rot_mat(unit_vec, plane_vec)
        fingertip_pos = np.mean(tool_repr, axis=0) - fingertip_mat @ tool_center_list[0]
        fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
        fingertip_T_list.append(fingertip_T)
    else:
        raise NotImplementedError

    return fingertip_T_list


def revise_tool_repr(args, tool_list, state, roller_rot_mat=None, visualize=False):
    tool_repr = state[args.n_particles+args.floor_dim:]
    dough_points = state[:args.n_particles+args.floor_dim]

    tool_repr_new = []
    tool_list_T = []
    fingertip_T_list = get_finger_T_list(args, tool_repr)
    for i in range(len(fingertip_T_list)):
        fingertip_T = fingertip_T_list[i]
        if roller_rot_mat is not None:
            tool_mesh_T = copy.deepcopy(tool_list[i][0]).rotate(roller_rot_mat, center=args.tool_center[args.env][i])
            tool_surface_T = copy.deepcopy(tool_list[i][1]).rotate(roller_rot_mat, center=args.tool_center[args.env][i])
            tool_mesh_T = tool_mesh_T.transform(fingertip_T)
            tool_surface_T = tool_surface_T.transform(fingertip_T)
        else:
            tool_mesh_T = copy.deepcopy(tool_list[i][0]).transform(fingertip_T)
            tool_surface_T = copy.deepcopy(tool_list[i][1]).transform(fingertip_T)
        
        tool_list_T.append((tool_mesh_T, tool_surface_T))
        tool_repr_new.append(np.asarray(tool_surface_T.points))

    if visualize:
        dough_pcd = o3d.geometry.PointCloud()
        dough_pcd.points = o3d.utility.Vector3dVector(dough_points)
        visualize_o3d([dough_pcd, *list(zip(*tool_list_T))[1]], title='transformed_tool_mesh')

    state_new = np.concatenate((state[:args.n_particles+args.floor_dim], *tool_repr_new), axis=0)

    return state_new


def main():
    args = gen_args()

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", "gt")
    src_dir = os.path.join(data_root_dir, f"data_{'_'.join(args.tool_type.split('_')[:-1])}")

    tool_name_list = args.tool_geom_mapping[args.env]
    if 'press' in args.env or 'punch' in args.env:
        tool_name_list.extend(args.tool_geom_mapping[args.env.split('_')[0] + '_circle'])

    tool_list_all = []
    for i in range(len(tool_name_list)):
        tool_mesh = o3d.io.read_triangle_mesh(os.path.join(args.tool_repr_path, f'{tool_name_list[i]}.stl'))
        tool_surface_dense = o3d.geometry.TriangleMesh.sample_points_uniformly(tool_mesh, 100000)

        if 'press_circle' in tool_name_list[i]:
            voxel_size = 0.0057
        else:
            voxel_size = 0.006

        tool_surface = tool_surface_dense.voxel_down_sample(voxel_size=voxel_size)
        
        if 'press_circle' in tool_name_list[i]:
            square_size = 184
            fps_points = fps(np.asarray(tool_surface.points), square_size)
            tool_surface = o3d.geometry.PointCloud()
            tool_surface.points = o3d.utility.Vector3dVector(fps_points)

        tool_surface.paint_uniform_color([1, 0, 0])
        print(f'{i}: {len(tool_surface.points)}')
        tool_list_all.append((tool_mesh, tool_surface))

    dataset_list = ["train", "valid", "test"]
    for dataset in dataset_list:
        dp_dir_list = sorted(glob.glob(os.path.join(src_dir, dataset, '*')))
        for i in range(0, len(dp_dir_list)):
            dest_dir = os.path.join(data_root_dir, f'data_{args.tool_type}', dataset, str(i).zfill(3))
            os.system(f"mkdir -p {dest_dir}")

            # for rollers
            roller_motion_z_dist_prev = 0
            spread = False
            state_prev = None
            roll_angle = 0
            roller_rot_mat = None

            dp = dp_dir_list[i]
            frame_list = sorted(glob.glob(os.path.join(dp, '*.h5')))
            state_seq = []
            for j in tqdm(range(len(frame_list))):
                frame = frame_list[j]
                frame_data = load_data(args.data_names, frame)
                state = frame_data[0]
                if 'roller' in args.env and state_prev is not None:
                    roller_motion = np.mean(state[args.n_particles+args.floor_dim:], axis=0) - \
                        np.mean(state_prev[args.n_particles+args.floor_dim:], axis=0)
                    roller_motion_z_dist = abs(roller_motion[2])
                    # print(roller_motion_z_dist)
                    if not spread and roller_motion_z_dist_prev > 0.0001 and roller_motion_z_dist < 0.0001:
                        print('spread!')
                        spread = True

                    roller_motion_z_dist_prev = roller_motion_z_dist

                    if spread:
                        roller_motion_xy_dist = np.linalg.norm(roller_motion[:2])
                        if roller_motion_xy_dist > 0:
                            roll_norm = np.cross(roller_motion[:2], (state[-1] - state[args.n_particles+args.floor_dim]))
                            roll_dir = roll_norm[2] / abs(roll_norm[2])
                            if 'large' in args.env:
                                roll_angle += roll_dir * roller_motion_xy_dist / 0.02
                            else:
                                roll_angle += roll_dir * roller_motion_xy_dist / 0.012

                        roller_rot_mat = axangle2mat([1, 0, 0], roll_angle)
                
                if 'press' in args.env or 'punch' in args.env:
                    unit_size = np.linalg.norm(state[args.n_particles+args.floor_dim+1] - state[args.n_particles+args.floor_dim])
                    # print(unit_size)
                    if unit_size > 0.004:
                        tool_list = [tool_list_all[0]]
                    else:
                        tool_list = [tool_list_all[1]]
                else:
                    tool_list = tool_list_all

                visualize = False
                state_new = revise_tool_repr(args, tool_list, state, roller_rot_mat=roller_rot_mat, visualize=visualize)
                state_seq.append(state_new)
                h5_data = [state_new, frame_data[1], frame_data[2]]
                store_data(args.data_names, h5_data, os.path.join(dest_dir, str(j).zfill(3) + '.h5'))
                
                state_prev = state

            render_anim(args, ['Perception'], [np.array(state_seq)], path=os.path.join(dest_dir, 'repr.mp4'))

            if args.debug: return


if __name__ == "__main__":
    main()
