plan_tool=gripper_sym_rod
n_particles=300
n_actions=2
early_fusion=1
use_normals=1
cls_weight=1.0
orient_weight=1.0
n_bin_rot=32
n_bin=8
rot_aug_max=0.25
train_set_ratio=1.0
debug=0

bash ./planning/policy/train.sh $plan_tool $n_particles $n_actions $early_fusion $use_normals $cls_weight $orient_weight $n_bin_rot $n_bin $rot_aug_max $train_set_ratio $debug
