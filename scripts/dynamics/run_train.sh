tool_type="gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16"
data_type="keyframe"   # default is keyframe, no special meaning
debug=0   # 1 for debug mode, in which case only 1 epoch is run
loss_type="chamfer_emd"   # a mix of chamfer and emd loss
n_his=1   # default is 1
sequence_length=2   # corresponds to s in Equation 4, Section 3.2 of the paper
time_step=3   # corresponds to t in Section 6.2.2 of the paper
rigid_motion=1   # 1 for consider both rigid and non-rigid motion, 0 for non-rigid motion only
attn=0   # default is 0, unsuccesful attempt to use attention
chamfer_weight=0.5
emd_weight=0.5
neighbor_radius=0.01   # the radius to connect edges in the graph, discussed in Section 6.2.1 of the paper
tool_neighbor_radius="default"
train_set_ratio=1.0

bash ./scripts/dynamics/train.sh $tool_type $data_type $debug $loss_type $n_his $sequence_length $time_step $rigid_motion $attn $chamfer_weight $emd_weight $neighbor_radius $tool_neighbor_radius $train_set_ratio
