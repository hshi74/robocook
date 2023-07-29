#!/usr/bin/env bash

tool_type="gripper_sym_plane_robot_v4_surf_nocorr_full"
dy_model_path="/scr/hshi74/projects/robocook/dump/dynamics/dump_gripper_sym_plane_robot_v4_surf_nocorr_full_normal_time_step=5/dy_synth_nr=0.01_tnr=0.004_0.004_his=1_seq=2_time_step=3_mse_rm=1_valid_Oct-14-23:31:24/net_best.pth"
n_rollout=1

bash ./scripts/dynamics/eval.sh $tool_type $dy_model_path $n_rollout
