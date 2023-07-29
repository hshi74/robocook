import os

plan_tool = "gripper_sym_rod"
n_particles = 300
n_actions = 2
early_fusion = 1
use_normals = 1
cls_weight = 1.0
orient_weight = 1.0
n_bin_rot = 32
n_bin = 8
rot_aug_max = 0.25
train_set_ratio = 0.1
debug = 0


def main():
    for tool in ["gripper_sym_rod", "gripper_asym", "gripper_sym_plane"]:
        for n_act in [1, 2]:
            for n_bin_rot_, n_bin_ in [(64, 8)]:
                param_str = f"{tool} {n_particles} {n_act} {early_fusion} {use_normals} {cls_weight} {orient_weight} {n_bin_rot_} {n_bin_} {rot_aug_max} {train_set_ratio} {debug}"
                print(param_str)
                os.system(f"sbatch ./control/policy/train.sh {param_str}")

            for rot_aug_max_ in [0.5, 1.0]:
                param_str = f"{tool} {n_particles} {n_act} {early_fusion} {use_normals} {cls_weight} {orient_weight} {n_bin_rot} {n_bin} {rot_aug_max_} {train_set_ratio} {debug}"
                print(param_str)
                os.system(f"sbatch ./control/policy/train.sh {param_str}")


if __name__ == "__main__":
    main()
