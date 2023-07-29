#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

python planning/policy/train.py \
	--plan_tool $1 \
	--n_particles $2 \
	--n_actions $3 \
	--early_fusion $4 \
	--use_normals $5 \
	--cls_weight $6 \
	--orient_weight $7 \
	--n_bin_rot $8 \
	--n_bin $9 \
	--rot_aug_max ${10} \
	--train_set_ratio ${11} \
	--debug ${12}