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
        print(f'{os.path.basename(data_dir)}: ')
        dataset_list = ["train", "valid", "test"]
        for dataset in dataset_list:
            data_point_dir_list = sorted(glob.glob(os.path.join(data_dir, dataset, '*')))
            n_frames = 0
            for data_point_dir in data_point_dir_list:
                for file in os.listdir(data_point_dir):
                    if os.path.isfile(os.path.join(data_point_dir, file)) and 'h5' in file:
                        n_frames += 1
            
            print(f'\t{dataset} set size: {n_frames} frames and {round(n_frames * 0.1 / 60, 1)} minutes')


if __name__ == "__main__":
    main()
