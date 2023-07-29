#!/usr/bin/env python

import copy
import csv
import glob
import os
import shutil
import sys

from yaml import dump

def main():
    date = "Oct-30"
    if len(sys.argv) < 2:
        # tool_arr = ["gripper_sym_robot_v1.5", "gripper_asym_robot_v1.5", "roller_small_robot_v1.5", "press_square_robot_v1.5",
        #     "gripper_sym_robot_v1.5_surf", "gripper_asym_robot_v1.5_surf", "roller_small_robot_v1.5_surf", "press_square_robot_v1.5_surf"]
        tool_arr = ["punch_square_robot_v4_surf_nocorr_full_normal_keyframe=16"]
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        root = os.path.join(cd, "..", "..")
        inspect_dir = f'{root}/dump/dynamics/inspect/{date}'
        os.system(f"mkdir -p {inspect_dir}/anim")
        dump_dir_list = []
        for tool_type in tool_arr:
            dy_dir = f"{root}/dump/dynamics/dump_{tool_type}"

            dump_dir_list_glob = sorted(glob.glob(os.path.join(dy_dir, '*')))
            for dump_dir in dump_dir_list_glob:
                if os.path.exists(f"{dump_dir}/eval") and date in dump_dir and not 'debug' in dump_dir:
                    dump_dir_list.append(dump_dir)

    else:
        tool_type = sys.argv[1]

        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        root = os.path.join(cd, "..", "..")
        dy_dir = f"{root}/dump/dynamics/dump_{tool_type}"
        inspect_dir = f'{dy_dir}/inspect'

        os.system(f"mkdir -p {inspect_dir}/anim")
        dump_dir_list = []
        dump_dir_list_glob = sorted(glob.glob(os.path.join(dy_dir, '*')))
        for dump_dir in dump_dir_list_glob:
            if os.path.exists(f"{dump_dir}/eval") and date in dump_dir and not 'debug' in dump_dir:
                dump_dir_list.append(dump_dir)

    stats = []
    for dump_dir in dump_dir_list:
        test_name = '_'.join(dump_dir.split('/')[-2:])
        print(test_name)

        dataset_list = ["train", "valid", "test"]
        n_rollout = 3
        for dataset in dataset_list:
            for i in range(n_rollout):
                i = str(i).zfill(3)
                res_anim = f"{dump_dir}/eval/anim/{dataset}_{i}.mp4"
                if os.path.exists(res_anim):
                    shutil.copyfile(res_anim, f'{inspect_dir}/anim/{test_name}_{dataset}_{i}.mp4')

        shutil.copyfile(f'{dump_dir}/eval/eval.txt', f'{inspect_dir}/eval_{test_name}.txt')
        
        with open(f'{inspect_dir}/eval_{test_name}.txt', 'r') as log_file:
            row = {'name': test_name}
            log = log_file.readlines()
            i = 0
            while i < len(log):
                line = log[i]

                if 'train set' in line: row['dataset'] = 'train'
                if 'valid set' in line: row['dataset'] = 'valid'
                if 'test set' in line: row['dataset'] = 'test'

                if len(row) > 0 and 'chamfer' in line.lower():
                    row_prev = copy.deepcopy(row)
                    if 'synth' in test_name:
                        loss_name_list = ['chamfer_surf', 'chamfer_synth', 'emd_surf', 'emd_synth', 'h_surf', 'h_synth']
                    else:
                        loss_name_list = ['chamfer_surf', 'emd_surf', 'h_surf']
                    for loss in loss_name_list:
                        row['loss'] = loss
                        last_loss = log[i+1].strip().split(':')[1].strip().replace("(", "").replace(")", "").split(' +- ')
                        avg_loss = log[i+2].strip().split(':')[1].strip().replace("(", "").replace(")", "").split(' +- ')
                        row_copy = copy.deepcopy(row)
                        row['type'], row['mean'], row['std'] = 'last', last_loss[0], last_loss[1]
                        row_copy['type'], row_copy['mean'], row_copy['std'] = 'avg', avg_loss[0], avg_loss[1]
                        stats.append(row)
                        stats.append(row_copy)
                        row = copy.deepcopy(row_prev)
                        i += 3
                    row = {'name': test_name}
                    continue

                i += 1

    fields = ['name','dataset', 'loss', 'type', 'mean', 'std']
    with open(f'{inspect_dir}/metrics.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

if __name__ == "__main__":
    main()
