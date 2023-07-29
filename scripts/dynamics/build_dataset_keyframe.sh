#!/usr/bin/env bash

python scripts/dynamics/build_dataset_keyframe.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v4_surf_nocorr_full_normal \
	--debug 0
