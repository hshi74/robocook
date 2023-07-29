#!/usr/bin/env python

import copy
import csv
import glob
import os
import shutil
import sys

def main():
    tool_arr = ["gripper_sym_robot_v4_surf_nocorr"]

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    root = os.path.join(cd, "..", "..")
    date = "Sep-19"
    n_rollout = 20
    os.system(f"mkdir -p {root}/dump/dynamics/inspect/{date}")
    for tool_type in tool_arr:
        data_dir = f"{root}/data/dynamics/data_{tool_type}"
        dy_dir = f"{root}/dump/dynamics/dump_{tool_type}"

        dump_dir_list = sorted(glob.glob(os.path.join(dy_dir, '*')))
        for dump_dir in dump_dir_list:
            if os.path.exists(f"{dump_dir}/eval") and date in dump_dir and not 'debug' in dump_dir:
                dy_model_path = dump_dir.split(f'{tool_type}/')[-1]
                print(dy_model_path)
                os.system(f"sbatch ./scripts/dynamics/eval.sh {tool_type} {dy_model_path}/net_best.pth {n_rollout}")

if __name__ == "__main__":
    main()
