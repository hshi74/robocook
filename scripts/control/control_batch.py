import os

from string import ascii_uppercase

tool_type = "gripper_sym_rod_robot_v3.1_surf_nocorr"
debug = 0
tool_model_name = (
    "dy_nr=0.0125_tnr=0.01_0.01_seq=8_chamfer_emd_0.2_0.8_rm=1_attn=1_valid"
)
optim_algo = "CEM"
CEM_sample_size = 40
control_loss_type = "chamfer"
subtarget = 1
close_loop = 0


def main():
    # image_names = list(ascii_uppercase)[:13]
    image_names = list(ascii_uppercase)[13:]
    # image_names = ['A']
    for name in image_names:
        target_shape_name = f"alphabet/{name}"
        param_str = f"{tool_type} {debug} {tool_model_name} {target_shape_name} {optim_algo} {CEM_sample_size} {control_loss_type} {subtarget} {close_loop}"
        print(param_str)
        os.system(f"sbatch ./scripts/control/control.sh {param_str}")


if __name__ == "__main__":
    main()
