#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

kernprof -l dynamics/eval.py \
	--stage dy \
	--tool_type $1 \
    --dy_model_path $2 \
	--n_rollout $3
