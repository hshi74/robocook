#!/usr/bin/env bash
# surface_sample: surface sampling or full sampling

kernprof -l perception/sample.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v4 \
	--n_particles 300 \
	--surface_sample 1
