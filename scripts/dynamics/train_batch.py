import os

tool_type="punch_square_robot_v4_surf_nocorr_full_normal_keyframe=16"
data_type="keyframe"
debug=0
loss_type="chamfer_emd"
n_his=1
sequence_length=2
time_step=1
rigid_motion=0
attn=0
chamfer_weight=0.5
emd_weight=0.5
neighbor_radius=0.01
tool_neighbor_radius="default"
train_set_ratio=1.0

def main():
    #for s in [2, 3]:
    #    for t in [1, 3, 5, 7]:
    #        if s == 3 and t > 5:
    #            continue
    #        rigid_motion = 1 if 'gripper' in tool_type else 0
    #        param_str = f'{tool_type} {data_type} {debug} {loss_type} {n_his} {s} {t} {rigid_motion} {attn} {chamfer_weight} {emd_weight} {neighbor_radius} {tool_neighbor_radius} {train_set_ratio}'
    #        print(param_str)
    #        os.system(f'sbatch ./scripts/dynamics/train.sh {param_str}')

    for tr in [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]:        
        param_str = f'{tool_type} {data_type} {debug} {loss_type} {n_his} {sequence_length} {time_step} {rigid_motion} {attn} {chamfer_weight} {emd_weight} {neighbor_radius} {tr} {train_set_ratio}'
        print(param_str)
        os.system(f'sbatch ./scripts/dynamics/train.sh {param_str}')

if __name__ == "__main__":
    main()
