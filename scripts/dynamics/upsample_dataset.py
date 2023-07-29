import glob
import numpy as np
import os
import shutil

from perception.pcd_utils import *
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_utils import *


class PolicyDataset(Dataset):
    def __init__(self, args, phase, rollout_ratio=1.0):
        self.args = args
        self.phase = phase
        self.rollout_ratio = rollout_ratio
        self.data_dir = os.path.join(args.plan_dataf, phase)
        self.data_names = ["positions", "shape_quats", "scene_params"]

        print(f"Loading {phase} data...")
        self.point_set = self._get_data()


def upsample(state, n_particles, visualize=False):
    state_pcd = o3d.geometry.PointCloud()
    state_pcd.points = o3d.utility.Vector3dVector(state[:, :3])
    state_surf_mesh = poisson_mesh_reconstruct(state_pcd, visualize=visualize)
    state_upsample_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
        state_surf_mesh, n_particles
    )
    state = np.asarray(state_upsample_pcd.points)

    state_normals = get_normals(state[None])[0]
    state = np.concatenate((state, state_normals), axis=1)

    if visualize:
        state_upsample_pcd.normals = o3d.utility.Vector3dVector(state_normals)
        state_upsample_pcd.paint_uniform_color([0, 1, 0])
        visualize_o3d([state_pcd, state_upsample_pcd], title="in_surface_point_cloud")

    return state


@profile
def upsample_data(src_dir, dest_dir, n_particles):
    data_names = ["positions", "shape_quats", "scene_params"]
    shape_quats = np.zeros((9, 4), dtype=np.float32)
    scene_params = np.array([1, 1, 0])

    for phase in ["train", "valid", "test"]:
        seq_path_list = sorted(glob.glob(os.path.join(src_dir, phase, "*")))
        for i in tqdm(range(len(seq_path_list))):
            sample_path_list = sorted(glob.glob(os.path.join(seq_path_list[i], "*")))
            for j in range(len(sample_path_list)):
                if j >= 10:
                    break

                state_path_list = sorted(
                    glob.glob(os.path.join(sample_path_list[j], "*.h5"))
                )
                if len(state_path_list) < 3:
                    continue

                new_sample_path = sample_path_list[j].replace(src_dir, dest_dir)
                if (
                    os.path.exists(new_sample_path)
                    and len(glob.glob(os.path.join(new_sample_path, "*"))) > 3
                ):
                    continue

                os.system("mkdir -p " + new_sample_path)
                for k in range(len(state_path_list)):
                    state = load_data(data_names, state_path_list[k])[0]
                    new_state = upsample(state, n_particles, visualize=False)

                    new_state_data = [new_state, shape_quats, scene_params]
                    store_data(
                        data_names,
                        new_state_data,
                        os.path.join(new_sample_path, f"{str(k).zfill(3)}.h5"),
                    )

                shutil.copy(
                    os.path.join(sample_path_list[j], "param_seq.npy"), new_sample_path
                )
                if os.path.exists(os.path.join(sample_path_list[j], "data.png")):
                    shutil.copy(
                        os.path.join(sample_path_list[j], "data.png"), new_sample_path
                    )


def main():
    src_dir = "data/planning_data/data_gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16_action=2_p=300"
    n_particles = 1024
    dest_dir = src_dir.replace("300", str(n_particles))

    upsample_data(src_dir, dest_dir, n_particles)


if __name__ == "__main__":
    main()
