#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

kernprof -l dynamics/train.py \
	--stage dy \
	--tool_type $1 \
	--data_type $2 \
	--debug $3 \
	--loss_type $4 \
	--n_his $5 \
	--sequence_length $6 \
	--time_step $7 \
	--rigid_motion $8 \
	--attn $9 \
	--chamfer_weight ${10} \
	--emd_weight ${11} \
	--neighbor_radius ${12} \
	--tool_neighbor_radius ${13} \
	--train_set_ratio ${14} \
	--eval 1 \
	--n_rollout 10
