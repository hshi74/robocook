#!/usr/bin/env bash

python scripts/perception/add_normals.py \
	--stage perception \
	--data_type synthetic \
	--tool_type gripper_asym_robot_v4_surf_nocorr_full_normal_time_step=5 \
	--debug 0
