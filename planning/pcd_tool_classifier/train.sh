#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

python planning/pcd_tool_classifier/train.py \
	--n_particles $1 \
	--early_fusion $2 \
	--use_rgb $3 \
	--debug $4