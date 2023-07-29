#!/usr/bin/env python

import copy
import csv
import glob
import os
import shutil
import sys

def main():
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", "gt")
    data_dir_list = sorted(glob.glob(os.path.join(data_root_dir, '*')))
    for data_dir in data_dir_list: 
        if not os.path.isdir(data_dir)  or 'classifier' in data_dir: continue
        if not 'data_roller_large_robot_v4_surf_nocorr_full' in data_dir: continue
        print(f'{os.path.basename(data_dir)}: ')
        dataset_list = ["train", "valid", "test"]
        for dataset in dataset_list:
            data_point_dir_list = sorted(glob.glob(os.path.join(data_dir, dataset, '*')))
            os.system(f"mkdir -p {os.path.join(data_dir, dataset, 'inspect')}")
            for i, data_point_dir in enumerate(data_point_dir_list):
                os.system(f"cp {os.path.join(data_point_dir, 'repr.mp4')} {os.path.join(data_dir, dataset, 'inspect', f'{str(i).zfill(3)}.mp4')}")


if __name__ == "__main__":
    main()
