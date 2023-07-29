#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

# kernprof -l
kernprof -l planning/control.py \
	--stage control \
	--tool_type $1 \
	--debug $2 \
	--tool_model_name $3 \
	--active_tool_list $4 \
	--target_shape_name $5 \
	--optim_algo $6 \
	--CEM_sample_size $7 \
	--control_loss_type $8 \
	--subtarget $9 \
	--close_loop ${10} \
	--cls_type ${11} \
	--planner_type ${12} \

