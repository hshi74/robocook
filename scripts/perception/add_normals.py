import copy
import glob
import numpy as np
import os
import shutil
import sys

from perception.pcd_utils import *
from tqdm import tqdm
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *


def get_normals_from_state(args, state, visualize=False):
    state_normals_list = []

    dough_points = state[:args.n_particles]
    dough_normals = get_normals(dough_points[None])[0]
    state_normals_list.append(dough_normals)
    
    dough_pcd = o3d.geometry.PointCloud()
    dough_pcd.points = o3d.utility.Vector3dVector(dough_points)
    dough_pcd.normals = o3d.utility.Vector3dVector(dough_normals)

    state_normals_list.append(args.floor_normals)
    
    floor_pcd = o3d.geometry.PointCloud()
    floor_points = state[args.n_particles: args.n_particles+args.floor_dim]
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.normals = o3d.utility.Vector3dVector(args.floor_normals)

    tool_start = args.n_particles + args.floor_dim
    tool_pcd_list = []
    for k in range(len(args.tool_dim[args.env])):
        tool_dim = args.tool_dim[args.env][k]
        tool_points = state[tool_start:tool_start+tool_dim]
        tool_normals = get_normals(tool_points[None])[0]
        state_normals_list.append(tool_normals)

        tool_pcd = o3d.geometry.PointCloud()
        tool_pcd.points = o3d.utility.Vector3dVector(tool_points)
        tool_pcd.normals = o3d.utility.Vector3dVector(tool_normals)
        tool_pcd_list.append(tool_pcd)
        
        tool_start += tool_dim

    # import pdb; pdb.set_trace()
    if visualize:
        o3d.visualization.draw_geometries([dough_pcd, floor_pcd, *tool_pcd_list], point_show_normal=True)

    return np.concatenate(state_normals_list, axis=0)


def gen_data(args, src_dir, dest_dir, visualize=False):
    frame_list = sorted(glob.glob(os.path.join(src_dir, '*.h5')))
    state_seq = []
    for frame in frame_list:
        frame_data = load_data(args.data_names, frame)
        state = frame_data[0]

        state_normals = get_normals_from_state(args, state, visualize=visualize)
        state_new = np.concatenate([state, state_normals], axis=1)
        state_seq.append(state_new)
        h5_data = [state_new, frame_data[1], frame_data[2]]
        store_data(args.data_names, h5_data, os.path.join(dest_dir, os.path.basename(frame)))

    if os.path.exists(os.path.join(src_dir, 'repr.mp4')):
        shutil.copyfile(os.path.join(src_dir, 'repr.mp4'), os.path.join(dest_dir, 'repr.mp4'))


def main():
    args = gen_args()
    dataset_list = ["train", "valid", "test"]
    for dataset in dataset_list:
        if args.data_type == 'synthetic':
            synth_dir_list = sorted(glob.glob(os.path.join(args.dy_data_path, dataset, '*')))
            for i in tqdm(range(0, len(synth_dir_list)), desc=dataset):
                src_dir_list = sorted(glob.glob(os.path.join(synth_dir_list[i], '*')))
                for j in range(len(src_dir_list)):
                    src_dir = src_dir_list[j]
                    dest_dir = os.path.join(f'{args.dy_data_path}_normal', dataset, str(i).zfill(3), str(j).zfill(3))
                    os.system(f"mkdir -p {dest_dir}")
                    gen_data(args, src_dir, dest_dir, visualize=False)

                    # frame_list = sorted(glob.glob(os.path.join(dest_dir, '*.h5')))
                    # for k in range(len(frame_list) - 1, -1, -1):
                    #     frame = os.path.basename(frame_list[k])
                    #     idx = int(frame.split('.')[0])
                    #     os.system(f"mv {frame_list[k]} {os.path.join(os.path.dirname(frame_list[k]), f'{str(idx + j).zfill(3)}.h5')}")

                    if args.debug: return
        else:
            src_dir_list = sorted(glob.glob(os.path.join(args.dy_data_path, dataset, '*')))
            for i in tqdm(range(0, len(src_dir_list)), desc=dataset):
                src_dir = src_dir_list[i]
                dest_dir = os.path.join(f'{args.dy_data_path}_normal', dataset, str(i).zfill(3))
                os.system(f"mkdir -p {dest_dir}")
                gen_data(args, src_dir, dest_dir, visualize=False)

                if args.debug: return


if __name__ == "__main__":
    main()
